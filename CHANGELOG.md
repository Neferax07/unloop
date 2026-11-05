# Changelog

All notable changes to unloop. Format: Keep a Changelog 1.1; semver.

## [0.1.0] - 2026-11-05

### Added

- Core governor: period (p <= 16) and echo (bounded window) detectors,
  minimal demotion below the runner-up, budget halt (`src/lib.rs`).
- Finite-ring phase governor: damp 5p+5 (order 6), kick 7p+7 (order 9,
  Hull-Dobell full period), seeded from prompt density class and lane.
- Presets loose / balanced / tight; preset chosen from prompt lane.
- Prompt profile: Shannon entropy, printable and digit shares, set-bit
  density class; lane selection documented in docs/STATE-MACHINE.md.
- Single-pass top-two selection: sequential reference plus fuzz-pinned fast
  path (4,000 adversarial cases), AVX2 per-lane top-2 under
  `target-cpu=native`, in-register NaN detection with sequential fallback.
- C ABI under feature `ffi` (`src/ffi.rs`).
- CLI: `unloop replay`, `unloop bench`, `unloop selftest` (`src/main.rs`).
- Tests: detector fixtures, schedule determinism, ring-order proofs,
  head-to-head against CTRL repetition penalty and frequency penalty on a
  deterministic degenerate decoder, zero-allocation proof via counting
  allocator.
- CI: three-OS correctness matrix, native-CPU SIMD job, bench smoke,
  package dry-run.
