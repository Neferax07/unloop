# unloop

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mark-dark.png">
  <img src="assets/mark.png" alt="unloop mark" width="160">
</picture>

Repetition-loop governor for greedy LLM decoding. Detects the continuation token of a
loop in the token history, demotes that single logit below the runner-up, schedules
escalation on an affine state machine over Z9. No weights, no training, no
dependencies.

```sh
cargo add unloop
```

## CLI

```
unloop replay --history <file> --logits <file> [--preset P] [--prompt S] [--json]
unloop bench
unloop selftest
```

Fixture formats:

- `history`: u32 little-endian token ids, one per step
- `logits`: f32 little-endian frames, one frame per step; `vocab = file_len / (4 * steps)`

`replay` prints the intervention schedule: step, verdict, phase, damp and kick
counters. `--json` emits the same trace as JSON.

## Library

```rust
use unloop::{profile, Governor, Preset, Step};

let prof = profile(prompt_bytes);            // once per prompt
let mut gov = Governor::new(Preset::for_profile(&prof), &prof);
loop {
    let step = gov.review(&history, &mut logits);  // per token, in place
    match step {
        Step::Pass | Step::Damp => { /* sample at your temperature */ }
        Step::Kick(t)           => { /* sample at t */ }
        Step::Stop              => break,
    }
}
```

`review` takes the full token history and the current logits frame; it writes at most
one logit. `Governor` is `Copy`, state under 64 bytes, no per-token allocation.

## C ABI

Feature `ffi`:

```c
unloop*  unloop_new(uint8_t preset, uint8_t density_class, uint8_t lane);
uint8_t  unloop_review(unloop* g, const uint32_t* history, uint32_t hlen,
                       float* logits, uint32_t n, float* kick_temp);
void     unloop_free(unloop* g);
```

Step tags: 0 pass, 1 damp, 2 kick (temperature written to `kick_temp`), 3 stop.

## State machine

Presets (`src/lib.rs`, `Policy::of`):

| preset  | patience | max_damps | max_kicks | margin | kick_T | min_span | echo_len | window |
|---------|----------|-----------|-----------|--------|--------|----------|----------|--------|
| loose   | 4        | 64        | 16        | 0.25   | 0.9    | 12       | 8        | 256    |
| balanced| 3        | 32        | 8         | 0.5    | 1.0    | 8        | 6        | 192    |
| tight   | 2        | 16        | 4         | 1.0    | 1.1    | 6        | 5        | 160    |

Per-token decision (`Governor::review`):

1. `loop_period`: smallest p ≤ 16 such that the history tail repeats with period p
   over ≥ 2 full periods and `matched + p ≥ min_span`.
2. `echo_continuation`: last `echo_len` tokens recur earlier within `window` tokens;
   continuation is the token that followed the earlier occurrence.
3. Degenerate only when a continuation exists and the logits argmax equals it.
4. Period-2 (ABAB) waits for 2 consecutive swings, then damps. Longer periods:
   `patience` passes, then up to 3 damps, then a kick. Budgets exceeded → `Stop`.
5. Demotion: `logits[top] = min(logits[top], logits[runner_up] - margin)`. No other
   logit is written.

Phase governor, seeded `phase = (density_class + lane_factor) % 9`, lane factor 3
(prose) / 6 (dense); `density_class` is the prompt's set-bit count folded to 1..=9:

```
damp: phase = (phase * 5 + 5) % 9   // order 6
kick: phase = (phase * 7 + 7) % 9   // order 9, full period (Hull-Dobell: gcd(7,9)=1,
                                    // 6 ≡ 0 mod 3, 4 ∤ 9)
kick temperature = kick_T * (1 + phase / 18)
```

Prompt lane: dense when Shannon entropy > 6 bits/byte, or printable share < 0.85, or
digit share ≥ 0.5; else prose. Preset from lane: prose → balanced, dense → tight.

Full algebra: [docs/STATE-MACHINE.md](docs/STATE-MACHINE.md).

## Measured

i7-13620H, 10C/16T, Windows 11 26300, `target-cpu=native`, median of 20,000 steps,
timed region is `review` only:

| path | ns/step |
|---|---|
| active loop (16k vocab, 512-token history, both detectors + logits pass) | 2,600–3,300 |
| healthy (no continuation, 0 logits touched) | 100–200 |

Reproduce: `cargo run --release --bin unloop -- bench`. Portable builds without
`target-cpu=native` run the exact scalar path, slower.

Reference head-to-head (deterministic 8-token-vocab mock, attractor from step 20,
64 steps, seed 42; reproducible via `cargo test --release -- --nocapture`):

| config | loop in tail | tokens to break | cumulative L1 damage |
|---|---|---|---|
| greedy | yes | — | 0 |
| repetition penalty θ=1.2 | yes | — | 17.90 |
| repetition penalty θ=1.5 | yes, relapses after 1 token | 1 | 25.45 |
| unloop (balanced) | no | 9 | 6.70 |

θ=1.2 cannot break a loop whose runner-up is already in the history: dividing both
logits by the same θ preserves their order. θ=1.5 escapes once, the escape token
enters the history, is penalized too, and the attractor wins the step back. unloop
holds: the escape route is not penalized, and each re-formed loop is damped again.

## Correctness gates (CI-enforced)

- only the continuation logit is written; every other logit is byte-identical
- 0 allocations per token, counting allocator over mount + healthy + intervention runs
- schedule replays byte-exact from (preset, prompt profile)
- fast top-two path equals the sequential reference over 4,000 adversarial cases
  (duplicate maxima, ±0, ±inf, sorted, reversed, random)
- SHA-256 free; no network; no I/O in the library path

## Files

```
src/lib.rs    core (no_std): detectors, governor, phase ring
src/main.rs   CLI: replay, bench, selftest
src/ffi.rs    C ABI (feature "ffi")
docs/STATE-MACHINE.md   full transition table and algebra
tests/        behavior, fuzz-equality, allocation proofs
```

## Prior art

- Holtzman et al., The Curious Case of Neural Text Degeneration, ICLR 2020.
  https://arxiv.org/abs/1904.09751
- Keskar et al., CTRL: repetition penalty, 2019. https://arxiv.org/abs/1909.05858
- Zhu et al., Penalty decoding / self-reinforcement, 2023. https://arxiv.org/pdf/2310.14971
- LZ Penalty, 2025. https://arxiv.org/html/2504.20131v2

## License

MIT OR Apache-2.0.
