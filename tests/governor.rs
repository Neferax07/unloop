// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Neferax

//! Head-to-head harness: unloop vs standard repetition penalty (CTRL-style) on
//! a deterministic degenerate-decoder mock. Every assertion is exact; every
//! metric is printed for the README table.

use unloop::{profile, shannon, Governor, Lane, Outcome, Preset, Step};

// ---------- deterministic mock decoder ----------

/// xorshift64* — deterministic per-step noise.
struct Xs(u64);
impl Xs {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((self.0 >> 40) as f32) / 16_777_216.0 - 0.5 // [-0.5, 0.5)
    }
}

const VOCAB: usize = 8;

enum Mock {
    /// Healthy race (tokens 2/3) for `attractor_at` steps, then a hard
    /// attractor on token 4 with the runner-up on token 6 — unseen in the
    /// healthy history, so the CTRL penalty gets its BEST case (it only has
    /// to out-divide the attractor against an unpunished alternative).
    Attractor { attractor_at: usize, noise: Xs },
    /// Genuinely non-repetitive: the winner walks a non-periodic pattern.
    Varied { noise: Xs },
}

impl Mock {
    fn attractor(noise_seed: u64) -> Self {
        Mock::Attractor { attractor_at: 20, noise: Xs(noise_seed) }
    }
    fn logits(&mut self, step: usize) -> [f32; VOCAB] {
        let mut l = [1.0f32; VOCAB];
        match self {
            Mock::Attractor { attractor_at, noise } => {
                if step < *attractor_at {
                    l[2] = 3.5 + noise.f32() * 0.2;
                    l[3] = 3.45 + noise.f32() * 0.2;
                } else {
                    l[6] = 3.5; // runner-up, never emitted before step 20
                    l[4] = 5.1 + noise.f32() * 0.2; // the attractor
                }
            }
            Mock::Varied { noise } => {
                // aperiodic winner walk: (step*3 + (step/7)*5) mod 7 has no
                // period ≤ 16 in the emitted token sequence
                let w = 1 + ((step * 3 + (step / 7) * 5) % 7) as usize;
                l[w] = 3.5 + noise.f32() * 0.2;
            }
        }
        l
    }
}

/// Greedy decode with a raw per-step transform. Returns tokens + cumulative
/// L1 damage vs the undisturbed model distribution.
fn decode_raw(mock: &mut Mock, steps: usize, transform: &mut dyn FnMut(usize, &mut [f32; VOCAB])) -> (Vec<u32>, f64) {
    let mut history = Vec::new();
    let mut damage = 0f64;
    for s in 0..steps {
        let clean = mock.logits(s);
        let mut l = clean;
        transform(s, &mut l);
        damage += l1(&clean, &l);
        let best = (0..VOCAB).fold(0usize, |a, i| if l[i] > l[a] { i } else { a });
        history.push(best as u32);
    }
    (history, damage)
}

fn softmax(v: &[f32]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let ex: Vec<f64> = v.iter().map(|&x| (x as f64 - max).exp()).collect();
    let sum: f64 = ex.iter().sum();
    ex.into_iter().map(|e| e / sum).collect()
}

fn l1(a: &[f32], b: &[f32]) -> f64 {
    let (pa, pb) = (softmax(a), softmax(b));
    pa.iter().zip(&pb).map(|(x, y)| (x - y).abs()).sum()
}

/// Exact-loop detector: is the tail a ≥32-token constant or period-2 run?
fn loops(history: &[u32]) -> bool {
    let n = history.len();
    if n < 32 {
        return false;
    }
    let tail = &history[n - 32..];
    if tail.iter().all(|&t| t == tail[0]) {
        return true;
    }
    tail[0] != tail[1]
        && tail.windows(2).step_by(2).all(|w| w[0] == tail[0] && w[1] == tail[1])
        && tail.windows(2).skip(1).step_by(2).all(|w| w[0] == tail[1] && w[1] == tail[0])
}

/// CTRL-standard repetition penalty (Keskar et al. 2019; HF semantics: divide
/// positive / multiply negative, once per unique seen token per step).
fn repetition_penalty(history: &[u32], logits: &mut [f32], theta: f32) {
    let mut seen = [false; VOCAB];
    for &t in history {
        let t = t as usize;
        if t < VOCAB && !seen[t] {
            seen[t] = true;
            if logits[t] > 0.0 {
                logits[t] /= theta;
            } else {
                logits[t] *= theta;
            }
        }
    }
}

/// Greedy + unloop governor (full history available, as a real server has).
fn decode_governed(mock: &mut Mock, steps: usize) -> (Vec<u32>, f64, Vec<u8>, Vec<(u16, u16)>) {
    let mut gov = Governor::new(Preset::Balanced, &profile(b"plain prose prompt"));
    let mut history = Vec::new();
    let mut damage = 0f64;
    let mut phases = Vec::new();
    let mut counters = Vec::new();
    for s in 0..steps {
        let clean = mock.logits(s);
        let mut l = clean;
        let step = gov.review(&history, &mut l);
        damage += l1(&clean, &l);
        phases.push(gov.phase());
        counters.push(gov.counters());
        if matches!(step, Step::Stop) {
            break;
        }
        let best = (0..VOCAB).fold(0usize, |a, i| if l[i] > l[a] { i } else { a });
        history.push(best as u32);
    }
    (history, damage, phases, counters)
}

/// Penalty decode that tracks history properly.
fn decode_penalty(mock: &mut Mock, steps: usize, theta: f32) -> (Vec<u32>, f64) {
    let mut history = Vec::new();
    let mut damage = 0f64;
    for s in 0..steps {
        let clean = mock.logits(s);
        let mut l = clean;
        if history.len() >= 8 {
            repetition_penalty(&history, &mut l, theta);
        }
        damage += l1(&clean, &l);
        let best = (0..VOCAB).fold(0usize, |a, i| if l[i] > l[a] { i } else { a });
        history.push(best as u32);
    }
    (history, damage)
}

// ---------- 0. exactness of the fast top_two (fuzz vs sequential reference) ----------

/// Sequential reference — the original scan rule, byte-exact.
fn top_two_reference(logits: &[f32]) -> Option<(usize, usize)> {
    if logits.len() < 2 {
        return None;
    }
    let (mut a, mut b) = if logits[1] > logits[0] { (1, 0) } else { (0, 1) };
    for (i, &v) in logits.iter().enumerate().skip(2) {
        if v > logits[a] {
            b = a;
            a = i;
        } else if v > logits[b] {
            b = i;
        }
    }
    Some((a, b))
}

#[test]
fn top_two_fast_matches_reference_everywhere() {
    // Adversarial families: constants, near-ties, duplicate maxima, sorted,
    // reverse, NaN, -inf, huge/tiny magnitudes, plus seeded random.
    let mut g = Xs(0xC0FFEE);
    let mut case_n = 0usize;
    let mut check = |logits: &[f32]| {
        case_n += 1;
        if let Some((top, runner)) = top_two_reference(logits) {
            // drive the crate's fast path through REAL reviews until the
            // governor intervenes (patience delays the first demotion)
            let hist = vec![top as u32; 12]; // period-1 continuation = top
            let mut l = logits.to_vec();
            let mut gov = Governor::new(Preset::Balanced, &profile(b"fuzz"));
            let mut fired = false;
            let mut schedule = Vec::new();
            for _ in 0..8 {
                let step = gov.review(&hist, &mut l);
                schedule.push(format!("{step:?}"));
                if matches!(step, Step::Damp | Step::Kick(_)) {
                    fired = true;
                    break;
                }
                l.copy_from_slice(logits); // Pass touched nothing; restore anyway
            }
            assert!(
                fired,
                "case {case_n}: no intervention within 8 reviews — logits={logits:?} want=({top},{runner}) schedule={schedule:?} phase={} counters={:?}",
                gov.phase(),
                gov.counters()
            );
            // fast path demoted exactly logits[top] = min(top, runner - margin)
            let expect = logits[top].min(logits[runner] - 0.5);
            let ok = l[top].to_bits() == expect.to_bits()
                || (!expect.is_infinite() && (l[top] - expect).abs() < 1e-6);
            assert!(
                ok,
                "case {case_n}: demotion target mismatch top={top} runner={runner} got {} want {expect}",
                l[top]
            );
            for (i, v) in logits.iter().enumerate() {
                if i != top {
                    assert_eq!(l[i], *v, "case {case_n}: non-top logit {i} was touched");
                }
            }
        }
    };
    // constants
    check(&[5.0; 2]);
    check(&[5.0; 64]);
    // near-ties and duplicate maxima
    check(&[1.0, 1.0 + f32::EPSILON]);
    check(&[3.0, 9.0, 7.0, 9.0]);
    check(&[9.0, 1.0, 9.0, 9.0]);
    check(&[9.0, 9.0, 1.0, 9.0]);
    check(&[5.0, 7.0, 7.0]);
    check(&[7.0, 7.0, 5.0]);
    check(&[7.0, 5.0, 7.0]);
    check(&[5.0, 9.0, 7.0]);
    check(&[-0.0, 0.0]);
    check(&[0.0, -0.0]);
    check(&[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY]);
    check(&[1e-30, -1e30, 1e-30, 1e30]);
    // sorted / reverse
    let asc: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let desc: Vec<f32> = asc.iter().rev().copied().collect();
    check(&asc);
    check(&desc);
    // seeded random (values + duplicates + extremes)
    for _ in 0..4000 {
        let n = 2 + (g.next_u64() as usize) % 512;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let r = g.next_u64();
            let val = match r % 16 {
                0 => f32::NEG_INFINITY,
                1 => f32::INFINITY,
                2 => 0.0,
                _ => ((r >> 11) as f32 / (1u32 << 23) as f32 - 2.0) * 10.0,
            };
            v.push(val);
        }
        check(&v);
    }
    // NaN family must route to the sequential path and still demote identically
    let nan_v = vec![f32::NAN, 5.0, 3.0, f32::NAN, 1.0];
    let (t, r) = top_two_reference(&nan_v).expect("reference selects top/runner on NaN vector");
    let hist = vec![t as u32; 12];
    let expect = nan_v[t].min(nan_v[r] - 0.5);
    let mut l = nan_v.clone();
    let mut gov = Governor::new(Preset::Balanced, &profile(b"fuzz"));
    let mut fired = false;
    for _ in 0..8 {
        let step = gov.review(&hist, &mut l);
        if matches!(step, Step::Damp | Step::Kick(_)) {
            fired = true;
            break;
        }
        l.copy_from_slice(&nan_v); // Pass touched nothing; restore anyway
    }
    assert!(fired, "NaN case: no intervention within 8 reviews");
    let ok = l[t].to_bits() == expect.to_bits()
        || (!expect.is_infinite() && (l[t] - expect).abs() < 1e-6);
    assert!(ok, "NaN-path demotion mismatch: got {} want {expect}", l[t]);
    for (i, v) in nan_v.iter().enumerate() {
        if i != t {
            assert_eq!(l[i].to_bits(), v.to_bits(), "NaN case: non-top logit {i} touched");
        }
    }
    let _ = &mut g;
}

impl Xs {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

// ---------- 1. structural detection ----------

#[test]
fn detects_periods_exact() {
    assert_eq!(unloop::loop_period(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 8), None);
    assert_eq!(unloop::loop_period(&[9, 7, 7, 7, 7, 7, 7, 7, 7], 8), Some(1));
    assert_eq!(unloop::loop_period(&[9, 1, 2, 1, 2, 1, 2, 1, 2], 8), Some(2));
    assert_eq!(unloop::loop_period(&[5, 1, 2, 3, 1, 2, 3, 1, 2, 3], 8), Some(3));
    assert_eq!(unloop::loop_period(&[1, 2, 1, 2], 8), None); // below min_span
}

#[test]
fn detects_non_adjacent_echo() {
    let h = [1u32, 2, 3, 4, 9, 8, 7, 1, 2, 3];
    assert_eq!(unloop::echo_continuation(&h, 3, 64), Some(4));
    assert_eq!(unloop::echo_continuation(&h, 4, 64), None);
    assert_eq!(unloop::echo_continuation(&h, 3, 4), None); // outside window
}

// ---------- 2. schedule + determinism ----------

#[test]
fn loop_breaks_with_minimal_intervention_and_replays() {
    let hist = vec![4u32; 12];
    let run = || {
        let mut g = Governor::new(Preset::Balanced, &profile(b"Once upon a time"));
        let mut schedule = Vec::new();
        let mut phases = Vec::new();
        for _ in 0..8 {
            let base = [0.0f32, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]; // top=4, runner=1
            let mut l = base;
            let step = g.review(&hist, &mut l);
            schedule.push(step);
            phases.push(g.phase());
            if step != Step::Pass {
                // minimal-intervention invariant: ONLY the continuation moved
                for i in [0usize, 2, 3, 5, 6, 7] {
                    assert_eq!(l[i], base[i]);
                }
                assert_eq!(l[1], base[1], "runner untouched");
                assert!(l[4] < l[1], "continuation must fall below runner-up");
            }
        }
        (schedule, phases, g.counters())
    };
    let (s1, p1, c1) = run();
    let (s2, p2, c2) = run();
    assert_eq!((s1.clone(), p1.clone(), c1), (s2, p2, c2), "byte-exact replay");
    // balanced: patience 3 → Pass, Pass, Damp, Damp, Damp, Kick, Pass, Pass
    assert_eq!(s1[0], Step::Pass);
    assert_eq!(s1[1], Step::Pass);
    assert!(matches!(s1[2], Step::Damp));
    assert!(matches!(s1[5], Step::Kick(_)));
}

#[test]
fn abab_swing_needs_two_consecutive() {
    let mut g = Governor::new(Preset::Tight, &profile(b"t"));
    let hist = vec![1u32, 2, 1, 2, 1, 2, 1, 2];
    let mut l = vec![1.0f32, 5.0, 0.0, 0.0]; // top=1 = period-2 continuation
    assert_eq!(g.review(&hist, &mut l), Step::Pass); // first swing tolerated
    assert_eq!(g.review(&hist, &mut l), Step::Damp); // second consecutive → damp
}

#[test]
fn no_intervention_when_model_already_leaving() {
    let mut g = Governor::new(Preset::Tight, &profile(b"t"));
    let hist = vec![3u32; 20];
    for _ in 0..10 {
        let mut l = vec![1.0f32, 0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0]; // top=5 ≠ continuation
        assert_eq!(g.review(&hist, &mut l), Step::Pass);
    }
    assert_eq!(g.outcome(), Outcome::Clean);
}

#[test]
fn budget_exhaustion_halts() {
    let mut g = Governor::new(Preset::Tight, &profile(b"t"));
    let hist = vec![3u32; 20];
    let stopped = (0..200).any(|_| {
        let mut l = vec![1.0f32, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        g.review(&hist, &mut l) == Step::Stop
    });
    assert!(stopped);
    assert_eq!(g.outcome(), Outcome::Halted);
}

#[test]
fn phase_governor_orders_are_exact() {
    // damp: p → (5p+5) mod 9 — walk from 0 until return: exactly 6 steps
    let mut p = 0u8;
    let mut damp_seen = vec![p];
    loop {
        p = (p * 5 + 5) % 9;
        damp_seen.push(p);
        if p == 0 {
            break;
        }
    }
    assert_eq!(damp_seen, vec![0, 5, 3, 2, 6, 8, 0], "damp map order 6");
    // kick: p → (7p+7) mod 9 — walk from 0 until return: exactly 9 steps
    let mut q = 0u8;
    let mut kick_seen = vec![q];
    loop {
        q = (q * 7 + 7) % 9;
        kick_seen.push(q);
        if q == 0 {
            break;
        }
    }
    assert_eq!(kick_seen.len(), 10, "kick map order 9 (full period)");
    let mut all = kick_seen.clone();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 9, "kick covers every ring state exactly once");
}

#[test]
fn prompt_seeds_the_phase() {
    let prose = profile(b"Once upon a time in a quiet village");
    let dense = profile(b"3.14159265 2.718281828 1.41421356");
    assert_eq!(prose.lane, Lane::Prose);
    assert_eq!(dense.lane, Lane::Dense);
    let g1 = Governor::new(Preset::Balanced, &prose);
    let g2 = Governor::new(Preset::Tight, &dense);
    assert_ne!(g1.phase(), g2.phase(), "prompt structure seeds the ring");
    assert_eq!(g1.outcome(), Outcome::Clean);
    assert!(shannon(b"aaaa") < shannon(&[0u8, 1, 2, 3, 4, 5, 6, 7]));
}

#[test]
fn shannon_accuracy_pinned() {
    // the no_std log2 path is pinned against exact closed forms, never assumed
    assert!((shannon(b"aabb") - 1.0).abs() < 1e-9);
    let all: Vec<u8> = (0..=255u8).collect();
    assert!((shannon(&all) - 8.0).abs() < 1e-9);
    assert_eq!(shannon(b""), 0.0);
    assert!(shannon(b"aaaa").abs() < 1e-12);
}

// ---------- 3. head-to-head vs repetition penalty ----------

#[test]
fn head_to_head_vs_repetition_penalty() {
    let steps = 64;
    let attractor_at = 20;

    // baseline: pure greedy — must loop
    let (h0, _) = decode_raw(&mut Mock::attractor(42), steps, &mut |_, _| {});
    assert!(loops(&h0), "greedy must enter the attractor");

    // CTRL θ=1.2: 5.1/1.2 = 4.25 > runner 3.5 — must STILL loop
    let (h12, d12) = decode_penalty(&mut Mock::attractor(42), steps, 1.2);
    let p12_loops = loops(&h12);

    // CTRL θ=1.5: 5.1/1.5 = 3.4 < runner 3.5 — escapes for ONE token, then
    // RELAPSES: once the escape token (6) enters the history it is punished
    // too (3.5/1.5 = 2.33), handing the win back to the attractor (3.4).
    // The penalty cannot hold its fix; it can only flash it.
    let (h15, d15) = decode_penalty(&mut Mock::attractor(42), steps, 1.5);
    let p15_loops = loops(&h15);

    // unloop: structural detection + minimal intervention + halt guarantee
    let (hu, du, phases, counters) = decode_governed(&mut Mock::attractor(42), steps);
    let u_loops = loops(&hu);

    // tokens-to-break: first non-attractor token after the attractor opens
    let t_break = |h: &[u32]| {
        h.iter()
            .enumerate()
            .skip(attractor_at)
            .find(|(_, &t)| t != 4)
            .map(|(i, _)| i - attractor_at)
            .unwrap_or(usize::MAX)
    };

    println!("== HEAD-TO-HEAD (64 steps, attractor opens @20, vocab 8, deterministic mock) ==");
    println!("{:<26} {:>8} {:>16} {:>10}", "config", "loop?", "tokens-to-break", "L1 damage");
    println!("{:<26} {:>8} {:>16} {:>10.2}", "greedy", format!("{}", loops(&h0)), "∞", 0.0);
    println!("{:<26} {:>8} {:>16} {:>10.2}", "rep-penalty θ=1.2", format!("{p12_loops}"), format!("{}", t_break(&h12)), d12);
    println!("{:<26} {:>8} {:>16} {:>10.2}", "rep-penalty θ=1.5 (relapses)", format!("{p15_loops}"), format!("{}", t_break(&h15)), d15);
    println!("{:<26} {:>8} {:>16} {:>10.2}", "unloop (balanced)", format!("{u_loops}"), format!("{}", t_break(&hu)), du);
    println!("unloop phases[:12]   : {:?}", &phases[..12.min(phases.len())]);
    println!("unloop final counters: {:?} (damps, kicks)", counters.last());

    // the claims, asserted:
    assert!(loops(&h0), "greedy loops");
    assert!(p12_loops, "θ=1.2 must still loop (5.1/1.2 > runner 3.5)");
    assert!(p15_loops && t_break(&h15) == 1, "θ=1.5 flashes one token then relapses");
    assert!(!u_loops, "unloop must break the loop and hold");
    assert!(
        t_break(&hu) <= 12,
        "unloop breaks within patience+cycle, got {}",
        t_break(&hu)
    );
    // damage: unloop is silent on healthy steps; the always-on penalty is not
    assert!(
        du < d15,
        "unloop cumulative L1 ({du:.3}) must undercut always-on penalty ({d15:.3})"
    );
    assert!(counters.last().map(|c| c.0 >= 1).unwrap_or(false), "at least one damp fired");
}

#[test]
fn healthy_generation_untouched() {
    // No repetition anywhere → the governor is fully silent (zero damage);
    // the always-on penalty is not.
    let (_, d_unloop, _, _) = decode_governed(&mut Mock::Varied { noise: Xs(7) }, 40);
    let (_, d_pen) = decode_penalty(&mut Mock::Varied { noise: Xs(7) }, 40, 1.3);
    println!("healthy-run damage: unloop {d_unloop:.6} vs always-on penalty {d_pen:.6}");
    assert!(d_unloop < 1e-9, "unloop must be silent on healthy runs");
    assert!(d_pen > 1e-3, "always-on penalty must show measurable distortion");
}
