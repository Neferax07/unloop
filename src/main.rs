// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Neferax

//! unloop CLI: replay, bench, selftest. std-only; reads binary fixtures, no
//! dependencies. Formats documented in `unloop replay --help`.

use std::io::Write as _;
use std::process::ExitCode;

use unloop::{profile, Governor, Preset, Step};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("replay") => cmd_replay(&args[1..]),
        Some("bench") => cmd_bench(),
        Some("selftest") => cmd_selftest(),
        _ => {
            usage();
            ExitCode::from(2)
        }
    }
}

fn usage() {
    eprintln!(
        "unloop — repetition-loop governor for greedy LLM decoding\n\
         \n\
         usage:\n\
         \x20 unloop replay --history <file> --logits <file> [--preset P] [--prompt S] [--json]\n\
         \x20 unloop bench\n\
         \x20 unloop selftest\n\
         \n\
         fixture formats (replay):\n\
         \x20 history : u32 little-endian token ids, one per step\n\
         \x20 logits  : f32 little-endian frames, frame = one vocab; vocab = len/(4*steps)"
    );
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn read_bytes(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| die(&format!("{path}: {e}")))
}

fn read_u32s(path: &str) -> Vec<u32> {
    let raw = read_bytes(path);
    if raw.len() % 4 != 0 {
        die(&format!("{path}: size {} not a multiple of 4", raw.len()));
    }
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32s(path: &str) -> Vec<f32> {
    let raw = read_bytes(path);
    if raw.len() % 4 != 0 {
        die(&format!("{path}: size {} not a multiple of 4", raw.len()));
    }
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse_preset(s: &str) -> Preset {
    Preset::parse(s).unwrap_or_else(|| die("preset must be loose|balanced|tight"))
}

// ---------- replay ----------

fn cmd_replay(args: &[String]) -> ExitCode {
    let mut history_path = None;
    let mut logits_path = None;
    let mut preset = Preset::Balanced;
    let mut prompt = String::from("prompt");
    let mut json = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--history" => history_path = Some(it.next().unwrap_or_else(|| die("--history needs a path"))),
            "--logits" => logits_path = Some(it.next().unwrap_or_else(|| die("--logits needs a path"))),
            "--preset" => {
                preset = parse_preset(it.next().unwrap_or_else(|| die("--preset needs loose|balanced|tight")));
            }
            "--prompt" => prompt = it.next().unwrap_or_else(|| die("--prompt needs a string")).clone(),
            "--json" => json = true,
            other => die(&format!("unknown flag {other}")),
        }
    }
    let history_path = history_path.unwrap_or_else(|| die("--history is required"));
    let logits_path = logits_path.unwrap_or_else(|| die("--logits is required"));

    let history = read_u32s(history_path);
    let mut logits = read_f32s(logits_path);
    let steps = history.len();
    if steps == 0 {
        die("empty history");
    }
    if logits.len() % steps != 0 {
        die(&format!(
            "logits len {} not divisible by steps {}",
            logits.len(),
            steps
        ));
    }
    let vocab = logits.len() / steps;

    let prof = profile(prompt.as_bytes());
    let mut gov = Governor::new(preset, &prof);
    if json {
        println!("{{\"preset\":\"{}\",\"vocab\":{vocab},\"steps\":{steps},\"trace\":[", preset.name());
    } else {
        println!("== UNLOOP REPLAY ==  preset {}  vocab {vocab}  steps {steps}", preset.name());
        println!("{:>6} {:>8} {:>6} {:>6} {:>6}", "step", "verdict", "phase", "damps", "kicks");
    }

    let out = std::io::stdout();
    let mut out = out.lock();
    for i in 0..steps {
        let frame = &mut logits[i * vocab..(i + 1) * vocab];
        let step = gov.review(&history[..i], frame);
        let verdict = match step {
            Step::Pass => "pass",
            Step::Damp => "damp",
            Step::Kick(_) => "kick",
            Step::Stop => "stop",
        };
        let (d, k) = gov.counters();
        if json {
            let _ = writeln!(
                out,
                "{{\"step\":{i},\"verdict\":\"{verdict}\",\"phase\":{},\"damps\":{d},\"kicks\":{k}}}{}",
                gov.phase(),
                if i + 1 == steps { "" } else { "," }
            );
        } else {
            let _ = writeln!(out, "{i:>6} {verdict:>8} {:>6} {d:>6} {k:>6}", gov.phase());
        }
        if matches!(step, Step::Stop) {
            let _ = writeln!(out, "halted at step {i}");
            break;
        }
    }
    if json {
        println!("]}}");
    }
    ExitCode::SUCCESS
}

// ---------- bench ----------

fn cmd_bench() -> ExitCode {
    const VOCAB: usize = 16_384;
    const HIST: usize = 512;
    const ITERS: usize = 20_000;

    let mut g0 = 0xABCDu64;
    let mut pristine = vec![-2.0f32; VOCAB];
    for x in pristine.iter_mut().skip(1) {
        g0 ^= g0 >> 12; g0 ^= g0 << 25; g0 ^= g0 >> 27;
        g0 = g0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        *x = -2.0 + ((g0 >> 40) as f32) / 4_000_000.0;
    }
    pristine[4] = 6.0;
    pristine[9] = 3.0;
    let history = vec![4u32; HIST];
    let healthy: Vec<u32> = (0..HIST).map(|i| ((i * 37) % 900) as u32 + 10).collect();
    let prof = profile(b"Once upon a time in a quiet village");

    println!("== UNLOOP BENCH ==  vocab {VOCAB}  history {HIST}  iters {ITERS}");

    let mut durs = Vec::with_capacity(ITERS);
    let mut buf = pristine.clone();
    for _ in 0..ITERS {
        buf.copy_from_slice(&pristine);
        let mut gov = Governor::new(Preset::Balanced, &prof);
        let t = std::time::Instant::now();
        let _ = gov.review(&history, &mut buf);
        durs.push(t.elapsed());
    }
    durs.sort();
    println!("loop, worst    : {:>8.0} ns/step", durs[durs.len() / 2].as_secs_f64() * 1e9);

    let mut durs = Vec::with_capacity(ITERS);
    let mut gov = Governor::new(Preset::Balanced, &prof);
    for _ in 0..ITERS {
        let t = std::time::Instant::now();
        let _ = gov.review(&healthy, &mut buf);
        durs.push(t.elapsed());
    }
    durs.sort();
    println!("healthy        : {:>8.0} ns/step", durs[durs.len() / 2].as_secs_f64() * 1e9);
    ExitCode::SUCCESS
}

// ---------- selftest ----------

fn cmd_selftest() -> ExitCode {
    let mut n = 0;
    let mut fail = 0;
    macro_rules! check {
        ($name:expr, $cond:expr) => {{
            n += 1;
            if $cond {
                println!("ok {n} - {}", $name);
            } else {
                fail += 1;
                println!("not ok {n} - {}", $name);
            }
        }};
    }

    // ring: damp map p -> (5p+5) % 9 has order 6 from 0
    let mut p = 0u8;
    let mut damp_steps = 0;
    loop {
        p = (p * 5 + 5) % 9;
        damp_steps += 1;
        if p == 0 {
            break;
        }
    }
    check!("damp map order 6", damp_steps == 6);

    // ring: kick map p -> (7p+7) % 9 is a full 9-cycle from 0
    let mut q = 0u8;
    let mut seen = [false; 9];
    let mut kick_steps = 0;
    loop {
        q = (q * 7 + 7) % 9;
        seen[q as usize] = true;
        kick_steps += 1;
        if q == 0 {
            break;
        }
    }
    check!("kick map full period 9", kick_steps == 9 && seen.iter().all(|&s| s));

    // detectors: exact periods
    check!(
        "period detector p=1",
        unloop::loop_period(&[9, 7, 7, 7, 7, 7, 7, 7, 7], 8) == Some(1)
    );
    check!(
        "period detector p=3",
        unloop::loop_period(&[5, 1, 2, 3, 1, 2, 3, 1, 2, 3], 8) == Some(3)
    );
    check!(
        "period detector no false positive",
        unloop::loop_period(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 8).is_none()
    );

    // echo: non-adjacent continuation
    let h = [1u32, 2, 3, 4, 9, 8, 7, 1, 2, 3];
    check!("echo continuation", unloop::echo_continuation(&h, 3, 64) == Some(4));

    // minimal intervention: only the continuation logit moves
    let hist = vec![4u32; 12];
    let base = [0.0f32, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
    let mut gov = Governor::new(Preset::Balanced, &profile(b"selftest"));
    let mut fired = false;
    let mut untouched = true;
    for _ in 0..8 {
        let mut l = base;
        let step = gov.review(&hist, &mut l);
        if !matches!(step, Step::Pass) {
            fired = true;
            for (i, b) in base.iter().enumerate() {
                if i != 4 && l[i] != *b {
                    untouched = false;
                }
            }
            if !(l[4] < base[1]) {
                untouched = false;
            }
            break;
        }
    }
    check!("intervention fires on forced continuation", fired);
    check!("only the continuation logit is demoted", untouched);

    // determinism: identical runs produce identical schedules
    let run = || {
        let mut g = Governor::new(Preset::Balanced, &profile(b"selftest"));
        let mut s = Vec::new();
        for _ in 0..8 {
            let mut l = base;
            s.push(g.review(&hist, &mut l));
        }
        s
    };
    check!("schedule replays byte-exact", run() == run());

    if fail == 0 {
        println!("# {n} tests passed");
        ExitCode::SUCCESS
    } else {
        println!("# {fail} of {n} failed");
        ExitCode::FAILURE
    }
}
