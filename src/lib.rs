//! # unloop — weight-free repetition governor for LLM decoding
//!
//! Zero dependencies, `no_std`, ~200 lines. Detects degenerate continuation loops
//! *structurally* (exact periods and non-adjacent echoes in the emitted token
//! history), then applies the **minimal intervention** that flips the greedy
//! choice: demote exactly one logit to just below the runner-up.
//!
//! ## Why not repetition penalty?
//! Additive/multiplicative penalties (CTRL-style) distort probability mass on
//! *every* decoding step, even healthy ones, and never halt a runaway. unloop
//! touches **zero logits on healthy steps**, moves the minimum mass needed to
//! break a detected loop, and runs a bounded schedule with a hard halt.
//!
//! ## The Deterministic Finite-Ring Phase Governor
//! Escalation runs on an affine state machine over ℤ₉:
//!
//! - `phase = (phase * 5 + 5) mod 9` on damp (order 6)
//! - `phase = (phase * 7 + 7) mod 9` on kick (order 9, full period by the
//!   Hull–Dobell theorem, so successive kicks never repeat before exhausting
//!   all 9 states)
//!
//! The phase is seeded from the prompt's Hamming-weight density class, and the
//! kick temperature scales with the live phase. Same prompt + same history ⇒
//! the same schedule, byte-exact — a determinism guarantee sampling-based
//! heuristics cannot make.
//!
//! ## Mount anywhere
//! See the `ffi` feature (`src/ffi.rs`) for the C ABI shim, and the crate tests
//! for a head-to-head harness against standard repetition penalty.

#![cfg_attr(not(feature = "ffi"), no_std)]

#[cfg(feature = "ffi")]
pub mod ffi;

// ---------- presets ----------

/// Intervention budget presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Few interventions, high patience.
    Loose,
    /// Default for prose prompts.
    Balanced,
    /// Default for dense (numeric/encoded) prompts.
    Tight,
}

impl Preset {
    /// Parses a preset name: `loose`, `balanced`, `tight`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "loose" => Some(Preset::Loose),
            "balanced" => Some(Preset::Balanced),
            "tight" => Some(Preset::Tight),
            _ => None,
        }
    }

    /// Canonical name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Preset::Loose => "loose",
            Preset::Balanced => "balanced",
            Preset::Tight => "tight",
        }
    }

    /// Preset chosen from a prompt profile.
    #[must_use]
    pub fn for_profile(p: &Profile) -> Self {
        match p.lane {
            Lane::Prose => Preset::Balanced,
            Lane::Dense => Preset::Tight,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Policy {
    patience: u8,
    max_damps: u16,
    max_kicks: u16,
    margin: f32,
    kick_temperature: f32,
    min_span: usize,
    echo_len: usize,
    window: usize,
}

impl Policy {
    fn of(p: Preset) -> Self {
        match p {
            Preset::Loose => Policy {
                patience: 4, max_damps: 64, max_kicks: 16, margin: 0.25,
                kick_temperature: 0.9, min_span: 12, echo_len: 8, window: 256,
            },
            Preset::Balanced => Policy {
                patience: 3, max_damps: 32, max_kicks: 8, margin: 0.5,
                kick_temperature: 1.0, min_span: 8, echo_len: 6, window: 192,
            },
            Preset::Tight => Policy {
                patience: 2, max_damps: 16, max_kicks: 4, margin: 1.0,
                kick_temperature: 1.1, min_span: 6, echo_len: 5, window: 160,
            },
        }
    }
}

// ---------- prompt profile ----------

/// Prompt lane: picks the default preset and seeds the phase governor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// Ordinary prose.
    Prose,
    /// Numeric / high-entropy / encoded-looking input.
    Dense,
}

impl Lane {
    fn factor(self) -> u8 {
        match self {
            Lane::Prose => 3,
            Lane::Dense => 6,
        }
    }
}

/// Structural profile of the prompt, computed once at mount.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    /// Shannon entropy, bits/byte.
    pub entropy: f64,
    /// Share of printable ASCII bytes.
    pub printable: f64,
    /// Share of ASCII digits among non-whitespace bytes.
    pub digits: f64,
    /// Hamming-weight density class: the prompt's total set-bit count folded
    /// into 1..=9 (`x % 9`, 0 → 9).
    pub density_class: u8,
    pub lane: Lane,
}

/// Folds a count into the classes 1..=9.
fn fold(x: u64) -> u8 {
    match x % 9 {
        0 => 9,
        r => r as u8,
    }
}

/// Total set bits across a byte slice.
fn ones(bytes: &[u8]) -> u64 {
    bytes.iter().map(|b| b.count_ones() as u64).sum()
}

/// log2 for x > 0, no_std-safe: IEEE-754 decomposition (x = m·2^e, m ∈ [1,2),
/// halved into [√2/2·2, √2) to shrink the series argument) plus the exact
/// identity ln(m) = 2·atanh((m−1)/(m+1)) summed through z¹⁹. Error < 1e-13
/// relative across the practical range — pinned by tests, never assumed.
fn log2_f64(x: f64) -> f64 {
    debug_assert!(x > 0.0, "log2 defined for x > 0");
    let bits = x.to_bits();
    let mut e = (((bits >> 52) & 0x7FF) as i64) - 1023;
    let mut m = f64::from_bits((bits & !(0x7FFu64 << 52)) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut sum = 0.0;
    let mut zk = z;
    let mut k = 1u32;
    while k <= 19 {
        sum += zk / k as f64;
        zk *= z2;
        k += 2;
    }
    e as f64 + 2.0 * sum / core::f64::consts::LN_2
}

/// Shannon entropy in bits/byte.
#[must_use]
pub fn shannon(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let n = data.len() as f64;
    freq.iter()
        .filter(|&&f| f > 0)
        .map(|&f| {
            let p = f as f64 / n;
            -p * log2_f64(p)
        })
        .sum()
}

/// Profiles a prompt: entropy, byte census, density class, lane.
#[must_use]
pub fn profile(data: &[u8]) -> Profile {
    let n = data.len().max(1) as f64;
    let printable = data
        .iter()
        .filter(|b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        .count() as f64
        / n;
    let solid = data.iter().filter(|b| !b.is_ascii_whitespace()).count().max(1) as f64;
    let digits = data.iter().filter(|b| b.is_ascii_digit()).count() as f64 / solid;
    let entropy = shannon(data);
    let dense = entropy > 6.0 || printable < 0.85 || digits >= 0.5;
    Profile {
        entropy,
        printable,
        digits,
        density_class: fold(ones(data)),
        lane: if dense { Lane::Dense } else { Lane::Prose },
    }
}

// ---------- verdicts ----------

/// What the decoder should do for the next token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// No intervention; sample as usual.
    Pass,
    /// Logits were adjusted; sample as usual.
    Damp,
    /// Logits were adjusted; sample once at this temperature.
    Kick(f32),
    /// Budget exhausted — stop decoding.
    Stop,
}

/// Whole-generation outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// No intervention fired.
    Clean,
    /// The governor intervened; generation continued.
    Steered,
    /// Budget exhausted; generation halted.
    Halted,
}

impl Outcome {
    /// Canonical name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Clean => "clean",
            Outcome::Steered => "steered",
            Outcome::Halted => "halted",
        }
    }
}

// ---------- the governor ----------

/// The repetition governor. One instance per generation stream; state < 64 B.
#[derive(Clone, Copy, Debug)]
pub struct Governor {
    policy: Policy,
    repeat: u8,
    swing: u8,
    soft: u8,
    phase: u8,
    damps: u16,
    kicks: u16,
    halted: bool,
}

impl Governor {
    /// Mounts a governor. The phase governor is seeded from the prompt's
    /// density class and lane factor — the prompt itself picks the starting
    /// state, so the schedule is a pure function of (preset, prompt).
    #[must_use]
    pub fn new(preset: Preset, profile: &Profile) -> Self {
        Governor {
            policy: Policy::of(preset),
            repeat: 0,
            swing: 0,
            soft: 0,
            phase: (profile.density_class % 9 + profile.lane.factor()) % 9,
            damps: 0,
            kicks: 0,
            halted: false,
        }
    }

    /// Live phase of the finite-ring governor (0..8).
    #[must_use]
    pub fn phase(&self) -> u8 {
        self.phase
    }

    /// Counters since mount (for receipts).
    #[must_use]
    pub const fn counters(&self) -> (u16, u16) {
        (self.damps, self.kicks)
    }

    /// Whole-run outcome so far.
    #[must_use]
    pub fn outcome(&self) -> Outcome {
        if self.halted {
            Outcome::Halted
        } else if self.damps + self.kicks > 0 {
            Outcome::Steered
        } else {
            Outcome::Clean
        }
    }

    fn reset(&mut self) {
        self.repeat = 0;
        self.swing = 0;
        self.soft = 0;
    }

    /// Inspects the history and may adjust `logits` in place.
    pub fn review(&mut self, history: &[u32], logits: &mut [f32]) -> Step {
        let period = loop_period(history, self.policy.min_span);
        let continuation = period
            .map(|p| history[history.len() - p])
            .or_else(|| echo_continuation(history, self.policy.echo_len, self.policy.window));
        let Some(next) = continuation else {
            self.reset();
            return Step::Pass;
        };
        let Some((top, runner_up)) = top_two(logits) else {
            return Step::Pass;
        };
        if top != next as usize {
            // The model is already leaving the repetition.
            self.reset();
            return Step::Pass;
        }

        let step = if period == Some(2) {
            // ABAB swing: wait for two consecutive swings before intervening.
            self.swing = self.swing.saturating_add(1);
            if self.swing >= 2 {
                self.swing = 0;
                self.damp(top, runner_up, logits)
            } else {
                Step::Pass
            }
        } else {
            self.repeat = self.repeat.saturating_add(1);
            if self.repeat < self.policy.patience {
                Step::Pass
            } else if self.soft < 3 {
                self.soft += 1;
                self.damp(top, runner_up, logits)
            } else {
                self.repeat = 0;
                self.soft = 0;
                self.kick(top, runner_up, logits)
            }
        };
        if self.damps > self.policy.max_damps || self.kicks > self.policy.max_kicks {
            self.halted = true;
            return Step::Stop;
        }
        step
    }

    /// Minimal intervention: lower the continuation to just below the runner-up.
    fn demote(&self, top: usize, runner_up: usize, logits: &mut [f32]) {
        logits[top] = logits[top].min(logits[runner_up] - self.policy.margin);
    }

    /// Damp: demote and advance the phase by the order-6 affine map.
    fn damp(&mut self, top: usize, runner_up: usize, logits: &mut [f32]) -> Step {
        self.demote(top, runner_up, logits);
        self.damps += 1;
        self.phase = (self.phase * 5 + 5) % 9;
        Step::Damp
    }

    /// Kick: demote and advance the phase by the order-9 (full-period) affine
    /// map; kick temperature scales with the live phase so successive kicks
    /// differ deterministically.
    fn kick(&mut self, top: usize, runner_up: usize, logits: &mut [f32]) -> Step {
        self.demote(top, runner_up, logits);
        self.kicks += 1;
        self.phase = (self.phase * 7 + 7) % 9;
        Step::Kick(self.policy.kick_temperature * (1.0 + self.phase as f32 / 18.0))
    }
}

// ---------- detectors ----------

// ---------- SIMD top-2 (x86_64 + AVX2 builds) ----------

/// True when the crate itself was compiled with AVX2 — the SIMD path is live.
pub const SIMD_AVX2: bool = cfg!(all(target_arch = "x86_64", target_feature = "avx2"));

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod simd {
    use core::arch::x86_64::*;

    /// Top-2 values (max, second-with-multiplicity) over a non-NaN f32 slice in
    /// ONE linear pass: per-lane `(g1, g2)` accumulators folded with the
    /// 3-value second law `g2' = max(g2, min(g1, k))`. Top-2-of-multiset is
    /// arrival-order independent, so lane-parallel == sequential, exactly.
    /// `vmaxps`/`vminps` implement f32 VALUE order (±0 equal), which is precisely
    /// the reference's operator semantics; NaNs are pre-filtered by the caller.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled at compile time (cfg-gated above).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn top2_f32(src: &[f32]) -> Option<(f32, f32)> {
        let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut g1 = ninf;
        let mut g2 = ninf;
        let mut nan = _mm256_setzero_ps(); // lane-OR of unordered self-compares
        let mut chunks = src.chunks_exact(8);
        for ch in &mut chunks {
            let k = _mm256_loadu_ps(ch.as_ptr());
            nan = _mm256_or_ps(nan, _mm256_cmp_ps(k, k, _CMP_UNORD_Q));
            let n1 = _mm256_max_ps(g1, k);
            let n2 = _mm256_max_ps(g2, _mm256_min_ps(g1, k));
            g1 = n1;
            g2 = n2;
        }
        // any NaN lane → caller re-runs the sequential reference path
        if _mm256_movemask_ps(nan) != 0 {
            return None;
        }
        // horizontal fold of the 8 lanes (multiset merge, scalar — 7 steps)
        let mut o1 = [0f32; 8];
        let mut o2 = [0f32; 8];
        _mm256_storeu_ps(o1.as_mut_ptr(), g1);
        _mm256_storeu_ps(o2.as_mut_ptr(), g2);
        let mut a1 = o1[0];
        let mut a2 = o2[0];
        for i in 1..8 {
            // merge (a1, a2) with (o1[i], o2[i]): second = max of the three
            // non-largest values
            if o1[i] > a1 {
                a2 = a1;
                a1 = o1[i];
                if o2[i] > a2 {
                    a2 = o2[i];
                }
            } else {
                if o1[i] > a2 {
                    a2 = o1[i];
                }
                if o2[i] > a2 {
                    a2 = o2[i];
                }
            }
        }
        // remainder (≤ 7 elements, scalar; NaN-checked per element)
        for &v in chunks.remainder() {
            if v.is_nan() {
                return None;
            }
            if v > a1 {
                a2 = a1;
                a1 = v;
            } else if v > a2 {
                a2 = v;
            }
        }
        Some((a1, a2))
    }

    pub(super) use top2_f32 as top2_f32_simd;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use simd::top2_f32_simd as top2_values;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn top2_values(src: &[f32]) -> Option<(f32, f32)> {
    let mut g1 = f32::NEG_INFINITY;
    let mut g2 = f32::NEG_INFINITY;
    for &v in src {
        if v.is_nan() {
            return None;
        }
        if v > g1 {
            g2 = g1;
            g1 = v;
        } else if v > g2 {
            g2 = v;
        }
    }
    Some((g1, g2))
}

/// Indices of the largest and second-largest logits.
///
/// Exactness contract (fuzz-pinned against the scalar reference in tests):
/// `top` is the FIRST index of the maximum; `runner` is the first index
/// `j != top` whose value equals the maximum over all `j != top`. This is
/// provably identical to the sequential scan rule (the second register ratchets
/// monotonically to the max-over-j≠top, and its first occurrence is where the
/// last update fired), so the demotion target is byte-identical. Index scans
/// run on the order-preserving u32 keys — strict-ordering exact, ±0 included.
///
/// NaN vectors take the sequential reference path.
fn top_two(logits: &[f32]) -> Option<(usize, usize)> {
    if logits.len() < 2 {
        return None;
    }
    // NaN vectors take the sequential reference path (the SIMD pass detects
    // them itself, in-register — no separate scan of the logits slice)
    let (m1, m2) = match unsafe { top2_values(logits) } {
        Some(v) => v,
        None => return top_two_sequential(logits),
    };
    // pass 2: first index attaining the max value — exact for non-NaN, because
    // the sequential reference selects the first index that REACHES the max
    // under strict ordering, i.e. the first index whose value equals it
    let mut top = None;
    for (i, &v) in logits.iter().enumerate() {
        if v == m1 {
            top = Some(i);
            break;
        }
    }
    let top = top?;
    // pass 3: first index j != top attaining the runner-up value
    let mut runner = None;
    for (i, &v) in logits.iter().enumerate() {
        if i != top && v == m2 {
            runner = Some(i);
            break;
        }
    }
    runner.map(|r| (top, r))
}

/// Sequential reference: the exact scan rule, kept for the NaN path and for
/// the fuzz-equality harness in tests. Never called on the hot path.
fn top_two_sequential(logits: &[f32]) -> Option<(usize, usize)> {
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

/// Smallest period p ≤ 16 such that the tail of `h` repeats with period p over
/// at least `min_span` tokens and at least two full periods.
///
/// Bounded-scan rewrite: the admission condition `matched ≥ p ∧ matched+p ≥
/// min_span` is exactly `matched ≥ needed` with `needed = max(p, min_span−p)`,
/// so comparisons stop at `needed` — the return value is identical to an
/// unbounded scan by construction.
#[must_use]
pub fn loop_period(h: &[u32], min_span: usize) -> Option<usize> {
    let n = h.len();
    (1..=16usize.min(n / 2)).find(|&p| {
        let needed = p.max(min_span.saturating_sub(p));
        let max_cmp = (n - p).min(needed);
        let mut matched = 0usize;
        while matched < max_cmp && h[n - 1 - matched] == h[n - 1 - matched - p] {
            matched += 1;
        }
        matched >= p && matched + p >= min_span
    })
}

/// If the last `len` tokens occurred earlier within the last `window` tokens,
/// returns the token that followed the most recent earlier occurrence.
///
/// Prefilter rewrite: first-token compare before the full slice equality —
/// same descending-j order (most recent first), same return value.
#[must_use]
pub fn echo_continuation(h: &[u32], len: usize, window: usize) -> Option<u32> {
    let n = h.len();
    if len == 0 || n <= len {
        return None;
    }
    let tail = &h[n - len..];
    let lo = n.saturating_sub(window);
    // j + len < n keeps the continuation inside the history and excludes the tail itself.
    // 8-wide chunked descending scan: whole-chunk existence prefilter (LLVM
    // lowers the unrolled OR to a vector compare), precise descending check
    // only when the chunk holds a first-token candidate.
    let t0v = tail[0];
    let mut hi = n - len;
    while hi > lo {
        let lo_c = if hi - lo >= 8 { hi - 8 } else { lo };
        let e = hi;
        let any = if e - lo_c == 8 {
            h[e - 1] == t0v
                || h[e - 2] == t0v
                || h[e - 3] == t0v
                || h[e - 4] == t0v
                || h[e - 5] == t0v
                || h[e - 6] == t0v
                || h[e - 7] == t0v
                || h[e - 8] == t0v
        } else {
            (lo_c..e).any(|k| h[k] == t0v)
        };
        if any {
            let mut k = e;
            while k > lo_c {
                k -= 1;
                if h[k] == t0v && h[k..k + len] == *tail {
                    return Some(h[k + len]);
                }
            }
        }
        hi = lo_c;
    }
    None
}
