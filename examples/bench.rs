//! unloop bench: per-token review cost, worst case (active loop continuation →
//! both detectors active + full logits pass) and healthy path. The timed region
//! is `review()` ONLY — the pristine-logits restore happens outside timing, so
//! no clone cost pollutes the numbers.

use std::time::{Duration, Instant};

use unloop::{profile, Governor, Preset};

const VOCAB: usize = 16_384;
const HIST: usize = 512;
const ITERS: usize = 20_000;

fn make_logits(seed: u64) -> Vec<f32> {
    let mut g = seed;
    let mut v = vec![-2.0f32; VOCAB];
    for x in v.iter_mut().skip(1) {
        g ^= g >> 12;
        g ^= g << 25;
        g ^= g >> 27;
        g = g.wrapping_mul(0x2545_F491_4F6C_DD1D);
        *x = -2.0 + ((g >> 40) as f32) / 4_000_000.0;
    }
    v[4] = 6.0; // the attractor continuation
    v[9] = 3.0; // runner-up
    v
}

fn median_ns(durs: &mut [Duration]) -> f64 {
    durs.sort();
    durs[durs.len() / 2].as_secs_f64() * 1e9
}

fn main() {
    let history = vec![4u32; HIST]; // worst case: constant continuation
    let healthy: Vec<u32> = (0..HIST).map(|i| ((i * 37) % 900) as u32 + 10).collect();
    let prof = profile(b"Once upon a time in a quiet village");
    let pristine = make_logits(0xABCD);
    let mut buf = pristine.clone();

    println!("== UNLOOP BENCH ==  vocab {VOCAB}, history {HIST}, {ITERS} steps");
    println!("   timed region = review() only; logit restore outside timing");
    println!(
        "   simd path: lib avx2={} (cfg target_feature), example avx2={}",
        unloop::SIMD_AVX2,
        cfg!(target_feature = "avx2")
    );

    // loop path: governor freshly mounted per step, logits restored per step
    let mut durs = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        buf.copy_from_slice(&pristine); // untimed
        let mut g = Governor::new(Preset::Balanced, &prof); // untimed mount
        let t0 = Instant::now();
        let _ = g.review(&history, &mut buf);
        durs.push(t0.elapsed());
    }
    println!("review (loop, worst)   : {:>8.0} ns/step", median_ns(&mut durs));

    // healthy path: no continuation → detectors bail, logits untouched
    let mut g = Governor::new(Preset::Balanced, &prof);
    let mut durs = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let _ = g.review(&healthy, &mut buf);
        durs.push(t0.elapsed());
    }
    println!("review (healthy)       : {:>8.0} ns/step   (0 logits touched)", median_ns(&mut durs));

    // isolation: where does loop-path time live?
    let mut durs = Vec::with_capacity(ITERS);
    let mut sink = 0u32;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let mut m1 = f32::NEG_INFINITY;
        for &v in &buf {
            if v > m1 {
                m1 = v;
            }
        }
        sink ^= m1.to_bits(); // defeat dead-code elimination
        durs.push(t0.elapsed());
    }
    println!("[isolate] bare max pass: {:>8.0} ns      (16k strict-gt, sink {sink:08x})", median_ns(&mut durs));
    // vocab scaling: does review cost track the logits slice?
    for vocab in [16usize, 1024, 16_384] {
        let mut small = pristine[..vocab].to_vec();
        small[4.min(vocab - 1)] = 6.0;
        if vocab > 9 {
            small[9] = 3.0;
        }
        let hist_small = vec![4u32; HIST];
        let mut durs = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            small.copy_from_slice(&pristine[..vocab]); // untimed
            let mut g = Governor::new(Preset::Balanced, &prof); // untimed
            let t0 = Instant::now();
            let _ = g.review(&hist_small, &mut small);
            durs.push(t0.elapsed());
        }
        println!("[isolate] review vocab={vocab:<6}: {:>8.0} ns/step", median_ns(&mut durs));
    }

    // isolation: pure AVX2 probes over the same 64 KB buffer
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        unsafe fn probe_max(src: &[f32]) -> f32 {
            #[target_feature(enable = "avx2")]
            unsafe fn inner(src: &[f32]) -> f32 {
                let mut g = _mm256_set1_ps(f32::NEG_INFINITY);
                let mut chunks = src.chunks_exact(8);
                for ch in &mut chunks {
                    let k = _mm256_loadu_ps(ch.as_ptr());
                    g = _mm256_max_ps(g, k);
                }
                let mut o = [0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), g);
                let mut a = o[0];
                for i in 1..8 {
                    if o[i] > a { a = o[i]; }
                }
                for &v in chunks.remainder() { if v > a { a = v; } }
                a
            }
            unsafe { inner(src) }
        }
        unsafe fn probe_top2(src: &[f32]) -> f32 {
            #[target_feature(enable = "avx2")]
            unsafe fn inner(src: &[f32]) -> f32 {
                let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
                let mut g1 = ninf;
                let mut g2 = ninf;
                let mut chunks = src.chunks_exact(8);
                for ch in &mut chunks {
                    let k = _mm256_loadu_ps(ch.as_ptr());
                    let n1 = _mm256_max_ps(g1, k);
                    let n2 = _mm256_max_ps(g2, _mm256_min_ps(g1, k));
                    g1 = n1;
                    g2 = n2;
                }
                let mut o = [0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), g1);
                let mut a = o[0];
                for i in 1..8 { if o[i] > a { a = o[i]; } }
                for &v in chunks.remainder() { if v > a { a = v; } }
                a
            }
            unsafe { inner(src) }
        }
        let mut durs = Vec::with_capacity(ITERS);
        let mut sink = 0f32;
        for _ in 0..ITERS {
            let t0 = Instant::now();
            sink = unsafe { probe_max(&buf) };
            durs.push(t0.elapsed());
        }
        println!("[isolate] avx2 max pass : {:>8.0} ns   (sink {sink})", median_ns(&mut durs));
        let mut durs = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            sink = unsafe { probe_top2(&buf) };
            durs.push(t0.elapsed());
        }
        println!("[isolate] avx2 top2 pass: {:>8.0} ns   (sink {sink})", median_ns(&mut durs));
        println!("[isolate] sink disposal  : {}", sink as u32);
    }

    println!("context                : a 0.5B-param model forward pass is ~1-10 ms —");
    println!("                         review overhead is noise; correctness is the product.");
}
