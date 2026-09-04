# Reportable result analysis: S vs I0, S-random, and E

## Analysis contract

- **Unit:** one `(scenario_id, chunk_idx)` base-trace/sentence-position target, not one continuation. Here `scenario_id` names a sampled base reasoning trace, not a distinct scenario.
- **Paired sample:** 19 targets from 11 independently sampled reasoning traces for one blackmail scenario. The E/S-random file has 21 targets, but S is missing 14/86, 17/86; all S comparisons use the 19-target intersection.
- **Outcome:** LLM-only blackmail verdict. This matches the released Thought Branches `contains_blackmail` labels better than the local compound email/keyword/LLM gate.
- **I0 construction:** for target sentence `i`, use the 100 valid continuations in `chunk_(i+1)/solutions.json`, where sentence `i` remains in the prefix.
- **Aggregation:** average continuations within each target and arm, then compute paired target differences. 95% intervals are percentile bootstraps over targets (50,000 samples, seed 20260904).

## Finding 1: S against I0

Among terminal-complete continuations, mean target-level blackmail fell from **41.5% under I0 to 32.0% under S**, a paired difference of **-9.6 pp** (95% target-bootstrap CI **[-16.7 pp, -2.4 pp]**). S was lower at 13 of 19 targets and higher at 6; the exact sign test is **p=0.167**, while the exact sign-flip test for the mean magnitude is **p=0.021**. The magnitude-weighted result is directional, but the target directions are heterogeneous.

The unfiltered observed contrast is larger (**-13.4 pp**), but it is not a valid headline estimate: a stop string ended about one-third of generated continuations immediately before the action, whereas I0 nearly always ran to completion. Restricting to terminal-complete outputs reduces that mismatch, but completion is post-treatment, so the -9.6 pp estimate remains a sensitivity analysis rather than a clean causal effect.

## Finding 2: S against S-random and E

Across the same targets and using all 20 generated continuations per target and arm, S averaged **28.2%** blackmail versus **32.6%** for S-random: **-4.5 pp** (95% CI **[-9.7 pp, +0.8 pp]**, sign-test p=0.210). This is a small, uncertain difference, so the reportable results do **not** show a clear separation between the reconsideration vector and norm-matched random steering.

S also closely matches the direct-edit arm E: **28.2%** for S versus **29.5%** for E, a paired difference of **-1.3 pp** (95% CI **[-6.8 pp, +4.2 pp]**, sign-test p=0.629). On the measured outcome, S and E are effectively indistinguishable at this sample size.

## Interpretation and limitations

The strongest defensible story is therefore asymmetric: S is associated with less blackmail than the original-sentence I0 baseline, but it does not outperform the two generated controls. That weakens a mechanism-specific interpretation. The S-random control is especially important: because it receives the same intervention machinery without the reconsideration direction, the small S-minus-S-random contrast is consistent with much of the S/I0 shift coming from generic generation/intervention effects rather than reconsideration content alone.

This conclusion is limited by (1) the stop-string truncation mismatch against I0, (2) conditioning on terminal completion in the sensitivity estimate, (3) only 19 targets, with multiple targets from some base traces, (4) the two missing classified S targets, (5) one scenario, model, task, layer, and coefficient, and (6) E's style confound. The figure should be paired with randomly sampled qualitative examples and a manual audit of judge labels in the final write-up.
