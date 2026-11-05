# unloop

**Kill LLM repetition loops with a 200-line, weight-free governor — deterministic, replayable, mountable anywhere.**

No model. No training. No dependencies. Detects degenerate continuation loops
*structurally* in the emitted token history, then applies the **minimal intervention**
that flips the greedy choice: demote exactly one logit to just below the runner-up.

```toml
[dependencies]
unloop = "0.1"
```

---

## The problem

Likelihood-maximizing decoding degenerates into repetition loops
([Holtzman et al., ICLR 2020](https://arxiv.org/abs/1904.09751)). The standard fix —
additive repetition penalty (CTRL, [Keskar et al. 2019](https://arxiv.org/abs/1909.05858)) —
distorts probability mass on **every** decoding step, even healthy ones, and as this
crate's harness proves, it can **flash** a fix and then **relapse**.

## The Relapse Proof Table

Deterministic degenerate decoder (vocab 8), attractor opens at step 20, 64-step
generations, xorshift noise seed 42 — every number reproducible from
`cargo test --release -- --nocapture`:

| config | loop in tail? | tokens-to-break | cumulative L1 damage |
|---|---|---|---|
| greedy | **yes** | ∞ | 0.00 |
| repetition penalty θ=1.2 | **yes** | ∞ | 17.90 |
| repetition penalty θ=1.5 | **yes — relapses** | 1 | 25.45 |
| **unloop (balanced)** | **no — holds** | 9 | **6.70** |

Two structural findings the harness proves:

1. **θ=1.2 cannot break a loop whose runner-up was already emitted** — dividing both
   logits by the same θ preserves their order. Mathematically impotent there.
2. **θ=1.5 escapes for exactly one token, then relapses**: the escape token enters the
   history, gets punished too, and the attractor wins it back. The penalty cannot
   *hold* a fix — only flash one. unloop's demotion holds: the escape route is never
   penalized, and each re-formed loop gets damped again.

On healthy runs unloop is **provably silent** (zero logits touched, L1 damage < 1e-9,
CI-asserted) while the always-on penalty measurably distorts every step
(`healthy_generation_untouched`).

## The mechanism

1. **Detect before intervening** — two structural detectors on the token history:
   exact-period detection (p ≤ 16, bounded suffix match) and non-adjacent echo
   detection (bounded window). A step is degenerate only when a repetition exists
   **and** the model's top choice is exactly the continuation.
2. **Minimal intervention** — lower the continuation logit to `runner-up − margin`.
   One logit written; every other logit is byte-untouched (CI-enforced invariant).
3. **Deterministic Finite-Ring Phase Governor** — escalation (damp → damp → damp →
   kick → … → halt) runs on an affine state machine over ℤ₉:
   - `phase = (phase * 5 + 5) mod 9` on damp (order 6),
   - `phase = (phase * 7 + 7) mod 9` on kick (order 9 — full period by the
     [Hull–Dobell theorem](https://en.wikipedia.org/wiki/Linear_congruential_generator),
     so successive kicks cycle all 9 states before repeating),
   - seeded from the prompt's Hamming-weight density class; kick temperature scales
     with the live phase.

   Same prompt + same history ⇒ the same schedule, byte-exact. No sampling-based
   heuristic can offer this.

## Performance

Hot path, single thread, i7-13620H (10C/16T), Windows, `target-cpu=native`
(reproducible: `cargo run --release --example bench`):

| path | before | after | directive |
|---|---|---|---|
| review — worst case (active loop, 16k vocab, 512-token history) | 27,127 ns | **3,300 ns** | < 3,500 ✓ |
| review — healthy step (no continuation, 0 logits touched) | 7,459 ns | **200 ns** | < 500 ✓ |

The logits pass is one linear AVX2 reduction (`vmaxps`/`vminps` per-lane top-2
accumulators — 2 vector ops per 8 logits, NaN-detected in-register, no scratch
buffer, no materialization). Portable builds fall back to the exact scalar path
(CI proves both paths pass the full suite). **0 allocations on every path**
(CI-enforced by a counting allocator in `tests/no_alloc.rs`). Build with
`RUSTFLAGS="-C target-cpu=native"` to enable the SIMD path on your machine —
without it everything is correct, just slower. The forward pass of a
0.5B-parameter model is 1–10 ms — review overhead is noise; correctness is the
product.

## Mount anywhere

Rust:

```rust
use unloop::{profile, Governor, Preset, Step};

let profile = profile(prompt_bytes);            // once per prompt
let mut gov = Governor::new(Preset::for_profile(&profile), &profile);
loop {
    let step = gov.review(&history, &mut logits); // per token
    match step {
        Step::Pass | Step::Damp => sample(&mut logits, temperature),
        Step::Kick(t)           => sample(&mut logits, t),
        Step::Stop              => break,
    }
}
```

C ABI (feature `ffi`):

```c
unloop* g = unloop_new(1, density_class, lane);
uint8_t step = unloop_review(g, history, hlen, logits, n, &kick_temp);
```

Determinism receipts: `Governor::phase()` and `Governor::counters()` expose the
governor state for replay verification.

## What this is not

- Not a replacement for nucleus/temperature sampling — orthogonal; mounts beside it.
- Not a claim to eliminate all repetition — it flips greedy continuation loops with
  minimal mass movement, deterministically, and halts when budgets exhaust.
- Not trained, not tuned per model — three presets, one documented policy table.

## Prior art

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) —
  Holtzman et al., ICLR 2020 (degeneration + nucleus sampling)
- [CTRL: repetition penalty](https://arxiv.org/abs/1909.05858) — Keskar et al. 2019
- [Penalty decoding / self-reinforcement](https://arxiv.org/pdf/2310.14971) — Zhu et al. 2023
- [LZ Penalty](https://arxiv.org/html/2504.20131v2) — 2025 (closest active competitor)
- [Towards Understanding Neural Text Degeneration](https://arxiv.org/html/2310.10226v1) — 2023

MIT OR Apache-2.0. Zero dependencies. `no_std` core.
