// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Neferax

//! C ABI shim — mount unloop into any runtime (llama.cpp, vLLM logits
//! processors, custom servers). Enabled by the `ffi` feature.
//!
//! ```c
//! #include <stdint.h>
//! typedef struct unloop unloop;
//! unloop* unloop_new(uint8_t preset, uint8_t density_class, uint8_t lane);
//! // step: 0=pass 1=damp 2=kick(temp out) 3=stop
//! uint8_t unloop_review(unloop* g, const uint32_t* history, uint32_t hlen,
//!                       float* logits, uint32_t n, float* kick_temp);
//! void    unloop_free(unloop* g);
//! ```

use crate::{Governor, Lane, Preset, Profile, Step};

pub struct UnloopHandle(Governor);

/// preset: 0=loose 1=balanced 2=tight · lane: 0=prose 1=dense · density class
/// is clamped to 1..=9.
///
/// # Safety
/// Returns a handle to be released with `unloop_free`.
#[no_mangle]
pub unsafe extern "C" fn unloop_new(preset: u8, density_class: u8, lane: u8) -> *mut UnloopHandle {
    let preset = match preset {
        0 => Preset::Loose,
        1 => Preset::Balanced,
        _ => Preset::Tight,
    };
    let profile = Profile {
        entropy: 0.0,
        printable: 1.0,
        digits: 0.0,
        density_class: density_class.clamp(1, 9),
        lane: if lane == 1 { Lane::Dense } else { Lane::Prose },
    };
    Box::into_raw(Box::new(UnloopHandle(Governor::new(preset, &profile))))
}

/// Reviews one decoding step in place. Returns the step tag; kick temperature
/// is written to `kick_temp` when the tag is 2.
///
/// A null handle, or a null buffer paired with a nonzero length, is a caller
/// bug: review returns 3 (stop) and touches nothing. This keeps a bad caller
/// fail-closed instead of undefined.
///
/// # Safety
/// `history` must be valid for `hlen` reads and `logits` valid for `n`
/// reads+writes whenever their lengths are nonzero.
#[no_mangle]
pub unsafe extern "C" fn unloop_review(
    g: *mut UnloopHandle,
    history: *const u32,
    hlen: u32,
    logits: *mut f32,
    n: u32,
    kick_temp: *mut f32,
) -> u8 {
    if g.is_null() || (hlen > 0 && history.is_null()) || (n > 0 && logits.is_null()) {
        return 3; // Stop: fail closed on a broken caller
    }
    let handle = unsafe { &mut *g };
    let history = unsafe { core::slice::from_raw_parts(history, hlen as usize) };
    let logits = unsafe { core::slice::from_raw_parts_mut(logits, n as usize) };
    match handle.0.review(history, logits) {
        Step::Pass => 0,
        Step::Damp => 1,
        Step::Kick(t) => {
            if !kick_temp.is_null() {
                unsafe { *kick_temp = t };
            }
            2
        }
        Step::Stop => 3,
    }
}

/// # Safety
/// `g` must come from `unloop_new` and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn unloop_free(g: *mut UnloopHandle) {
    if !g.is_null() {
        drop(unsafe { Box::from_raw(g) });
    }
}
