# unloop state machine

Spec of every transition in `src/lib.rs`, with the function that implements
each rule. All arithmetic is integer except the logits demotion and the
Shannon entropy in the prompt profile.

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

Cycle structure (checked at every `unloop selftest` run, tests 1 and 2):

- damp: order 6 from 0, cycle 0 -> 5 -> 3 -> 2 -> 6 -> 8 -> 0. The map also
  has a fixed point at 1 and a 2-cycle (4 7), which no reachable schedule
  enters because the walk starts at the seed below and only uses these two
  maps from there.
- kick: order 9 from 0, a single cycle over all nine residues.

Hull-Dobell gives the full-period conditions for x -> a*x + b mod m. For the
kick map, m = 9, a = 7, b = 7: gcd(b, m) = 1 holds, (a - 1) = 6 is divisible
by the only prime factor 3 of 9, and the "4 divides m" clause is vacuous
since 4 does not divide 9. The damp map fails the second condition
(a - 1 = 4, not divisible by 3), which is exactly why its order collapses to
6.

## Seed

`Governor::new`, from the prompt profile and the lane factor:

```
density_class = dr(set_bit_count(prompt_bytes))
lane = Dense if shannon(prompt) > 6.0
         or printable_share < 0.85
         or digit_share >= 0.5
       else Prose
lane_factor = 3 (Prose) | 6 (Dense)
phase = (density_class + lane_factor) mod 9
```

Same prompt bytes and preset, same schedule. Pinned by
`tests/governor.rs::loop_breaks_with_minimal_intervention_and_replays`.

## Detectors

`loop_period(history, min_span)`: smallest p in 1..=16 with p <= len/2 such
that the tail matches at distance p for at least `needed` comparisons, where
`needed = max(p, min_span - p)`; accepted when matched >= p and
matched + p >= min_span. The comparison cap is an identity of the admission
rule, so the returned value equals an uncapped scan.

`echo_continuation(history, echo_len, window)`: take the last echo_len
tokens as the tail, scan positions j in descending order over the window,
and return `history[j + echo_len]` for the first j whose
`history[j..j+echo_len] == tail`. Descending order is what makes the most
recent earlier occurrence win.

## Decision rule

`Governor::review`, per token:

1. continuation = period map (token at distance p), else echo map
2. no continuation: reset the repeat/swing/soft counters, return Pass
3. (top, runner) = first argmax of the logits, then first index != top equal
   to the max over j != top. Fuzz-pinned to the sequential reference in
   `tests/governor.rs::top_two_fast_matches_reference_everywhere`
4. top != continuation: the model is leaving on its own; reset, return Pass
5. period == 2: swing counter += 1; on the second consecutive swing, damp
6. otherwise: repeat counter += 1; below patience, Pass; then up to 3 damps;
   then a kick and reset of repeat/soft
7. after any intervention: damps > max_damps or kicks > max_kicks sets
   halted and returns Stop. This is the only case where one event overrides
   the verdict computed in rule 5 or 6.

## Intervention

```
logits[top] = min(logits[top], logits[runner] - margin)
```

One write. CI-invariant: every other logit stays byte-identical
(`tests/governor.rs`, minimal-intervention check; `tests/no_alloc.rs` proves
zero allocation across mount, healthy runs, and intervention runs).

## Preset table

See README. The values live in one place, `Policy::of` in `src/lib.rs`.

## Byte budgets

- Governor state: the Copy policy plus repeat, swing, soft, phase, damps,
  kicks and a halted flag; under 64 B per stream
- history window: at most `window` (160 to 256) tokens inspected per step;
  the period detector is capped at 16 candidate periods of at most `needed`
  comparisons each
- logits pass: one linear scan. AVX2 builds fold per-lane top-2 with
  vmaxps/vminps; NaN is detected in-register and those frames route to the
  sequential reference path
