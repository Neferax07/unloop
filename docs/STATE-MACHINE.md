# unloop state machine

Spec of every transition in `src/lib.rs`. Each rule cites the function that
implements it. All arithmetic is integer except the logits demotion and the
Shannon-entropy lane profile.

## Ring

State space: residues of Z/9Z, displayed 1..=9 with residue 0 shown as 9.

```
dr(x) = x % 9, remainder 0 mapped to 9
```

Two affine endomorphisms drive the schedule:

```
damp: p -> (5p + 5) mod 9
kick: p -> (7p + 7) mod 9
```

Cycle structure (computed in `unloop selftest`, test 1 and 2):

- damp: order 6 from 0, cycle 0 -> 5 -> 3 -> 2 -> 6 -> 8 -> 0; the map has a
  fixed point at 1 and a 2-cycle (4 7)
- kick: order 9 from 0, single cycle covering all nine residues

Hull-Dobell full-period conditions for x -> a*x + b mod m, m = 9, kick map
a = 7, b = 7: gcd(b, m) = 1 holds; (a - 1) = 6 is divisible by the only prime
factor 3 of 9; 4 does not divide 9 so that clause is vacuous. The damp map
violates the second condition (a - 1 = 4 not divisible by 3), which forces its
reduced order.

## Seed

`Governor::new` (`profile` + lane factor):

```
density_class = dr(set_bit_count(prompt_bytes))
lane = Dense if shannon(prompt) > 6.0
         or printable_share < 0.85
         or digit_share >= 0.5
       else Prose
lane_factor = 3 (Prose) | 6 (Dense)
phase = (density_class + lane_factor) mod 9
```

Same prompt bytes and preset produce the same schedule. Verified in
`tests/governor.rs::loop_breaks_with_minimal_intervention_and_replays`.

## Detectors

`loop_period(history, min_span)`: smallest p in 1..=16 with
p <= len/2 such that the tail matches at distance p for at least `needed`
comparisons where `needed = max(p, min_span - p)`; accepted when matched >= p
and matched + p >= min_span. The comparison cap is an identity of the
admission rule; the returned value equals an uncapped scan.

`echo_continuation(history, echo_len, window)`: let tail be the last echo_len
tokens; scan j descending from len - echo_len - 1 down to len - window; return
history[j + echo_len] of the first j with history[j..j+echo_len] == tail.
Descending order makes the most recent earlier occurrence win.

## Decision rule

`Governor::review`, per token:

1. continuation = period map (token at distance p) else echo map
2. no continuation: reset repeat/swing/soft counters, return Pass
3. (top, runner) = first argmax of logits, then first index != top equal to
   max over j != top (fuzz-pinned to the sequential reference,
   `tests/governor.rs::top_two_fast_matches_reference_everywhere`)
4. top != continuation: model is leaving on its own; reset, return Pass
5. period == 2: swing counter += 1; on the second consecutive swing, damp
6. otherwise: repeat counter += 1; below patience, Pass; then up to 3 damps;
   then kick and reset repeat/soft
7. after any intervention: damps > max_damps or kicks > max_kicks sets halted,
   returns Stop
8. verdict precedence when several events fire in one step: BaselineSync-type
   primary events win over CycleOrbit-style secondary counters; counters
   update regardless

Verdict precedence applies to the tickgate engine; in unloop the only
multi-event case is Stop overriding the step computed in rule 5/6.

## Intervention

```
logits[top] = min(logits[top], logits[runner] - margin)
```

One write. CI-invariant: every other logit byte-identical
(`tests/governor.rs`, minimal-intervention check; `tests/no_alloc.rs` proves
zero allocation across mount, healthy runs, and intervention runs).

## Preset table

See README. Values live in one place: `Policy::of` in `src/lib.rs`.

## Byte budgets

- Governor state: preset policy (Copy) + 5 counters + phase + head digest
  slot; under 64 B per stream
- history window: at most `window` (160..256) tokens inspected per step;
  period detector cost bounded at 16 comparisons of at most `needed` length
- logits pass: single linear scan; AVX2 build folds per-lane top-2 with
  vmaxps/vminps; NaN detected in-register, NaN frames route to the sequential
  reference path
