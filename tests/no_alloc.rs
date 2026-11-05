// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Neferax

//! Allocation proof: the README claims "0 allocations on every path" — this
//! test makes that claim CI-enforced instead of asserted. Counting allocator
//! around healthy (Pass) and intervention (Damp/Kick) reviews plus mounts.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use unloop::{profile, Governor, Preset, Step};

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { unsafe { System.dealloc(p, l) } }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(n as u64, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn zero_allocations_on_every_path() {
    let prompt = b"Once upon a time in a quiet village";
    let vocab = 16_384usize;
    // build fixtures BEFORE the measured regions
    let mut g0 = 0x1234_5678_9ABC_DEF0u64;
    let mut logits = vec![-2.0f32; vocab];
    for x in logits.iter_mut().skip(1) {
        g0 ^= g0 >> 12; g0 ^= g0 << 25; g0 ^= g0 >> 27;
        g0 = g0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        *x = -2.0 + ((g0 >> 40) as f32) / 4_000_000.0;
    }
    logits[4] = 6.0;
    logits[9] = 3.0;
    let pristine = logits.clone();
    let history = vec![4u32; 512];
    let healthy: Vec<u32> = (0..512).map(|i| ((i * 37) % 900) as u32 + 10).collect();
    let prof = profile(prompt);

    // mount
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    let mut gov = Governor::new(Preset::Balanced, &prof);
    assert_eq!(ALLOC_COUNT.load(Ordering::Relaxed), 0, "mount allocated");

    // healthy path (Pass)
    for _ in 0..100 {
        let _ = gov.review(&healthy, &mut logits);
    }
    assert_eq!(ALLOC_COUNT.load(Ordering::Relaxed), 0, "healthy reviews allocated");

    // intervention path: drive to Damp/Kick on a looping history
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    let mut gov2 = Governor::new(Preset::Balanced, &prof);
    let mut fired = 0;
    for step_num in 0..40 {
        logits.copy_from_slice(&pristine);
        let step = gov2.review(&history, &mut logits);
        if matches!(step, Step::Damp | Step::Kick(_)) {
            fired += 1;
        }
        let _ = step_num;
    }
    assert!(fired >= 1, "harness sanity: interventions must fire");
    assert_eq!(
        ALLOC_COUNT.load(Ordering::Relaxed),
        0,
        "intervention reviews allocated ({} allocs)",
        ALLOC_COUNT.load(Ordering::Relaxed)
    );
    assert_eq!(ALLOC_BYTES.load(Ordering::Relaxed), 0);
}
