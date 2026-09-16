# Consolidation certificate and replay audit

Recorded on 2026-09-15. Historical results were preserved; this directory contains
new diagnostic evidence. The v2 campaign remains complete at **640 attempts and
135 recorded successes**, but the missing original FO strategies prevent its
full recertification. Its bounded three-seed replay has **five regret mismatches
and one success-classification change**. Treat the original figure as recorded
outcomes with this unresolved reproducibility limitation.

## Available profiles

Fresh full-sample best-response LPs agree with all **3,333** comparable recorded
certificates: 3,200 fixed uniform profiles and 133 saved calibration profiles.
Successful and failed returned profiles were both included. These counts exceed
the displayed README cells because the whole baseline dataset was checked.
An additional 1,100 archived notebook profiles were repriced, but their strategies
were rounded to three decimals and no original regret was recorded; they cannot
establish agreement with the original full-precision solves.

| Study/run | Row-level audit coverage |
| --- | --- |
| scalability/msd_cvar_original | intentionally_absent: 571; missing_profile: 2309 |
| scalability/cvar_capped_24h_v1 | missing_profile: 2178; intentionally_absent: 702 |
| scalability/equal_screen_v1 | intentionally_absent: 133; missing_profile: 107 |
| qptas_scalability/sampled_1000_v1 | missing_profile: 537; intentionally_absent: 423 |
| population_qptas_fo/beta_uniform_v2 | intentionally_absent: 320; missing_profile: 320 |
| uniform_baseline/v1 | match: 3200 |
| stochastic_fo/continuation_v2 | missing_profile: 36 |
| stochastic_fo/continuation_v3 | missing_profile: 36 |
| population_fo_calibration/pilot_v1 | missing_profile: 100; match: 60 |
| population_fo_calibration/cvar_steps_v2 | match: 73 |
| archive/notebook_results | no_recorded_regret: 1100 |

`missing_profile` means the saved output indicates a returned result whose final
strategy is unavailable. `intentionally_absent` denotes exhausted searches or
failed attempts without a returned certificate; a missing QPTAS eta is never
replaced with zero. Uniform strategies are known by construction. Missing original
payoff-array hashes make these regenerated-game checks **conditional**. The audit
uses the existing package LPs plus independently evaluated state payoffs and risk
payoffs; it is not an independent LP implementation. Numeric comparisons use
atol=1e-8 and rtol=1e-6, and also check the unrounded eta <= 0.01 classification
separately. Simplex validity and LP failure are checked before accepting agreement.
Some local NumPy matrix-product warnings were observed during checking; they did
not produce non-finite certificates or independent-payoff discrepancies.

## Missing-profile replays

The selection was frozen before execution using RNG seed 0. It prefers the
smallest presented `(n, K)` games, includes both risks where represented, chooses
three distinct recorded game seeds per study/run, and never selects by eta or
success. All affected method/configuration rows for each selected seed share a
100-second budget. A study has a 300-second aggregate budget, including child
startup; the process group is killed at the deadline. Observed cleanup/scheduling
overhead was at most a few hundredths of a second. No solver iteration budget was
reduced, no seed was replaced, and no full campaign was rerun.

| Study/run | Frozen game seeds | Aggregate seconds | Method/configuration comparisons |
| --- | --- | ---: | --- |
| population_fo_calibration/pilot_v1 | 15128123, 14128123 | 192.12 | match: 84; timeout_or_unattempted: 16 |
| population_qptas_fo/beta_uniform_v2 | 6005140, 5005135, 6005132 | 81.23 | mismatch: 5; classification_flip: 1 |
| qptas_scalability/sampled_1000_v1 | 1050630, 50628, 1050626 | 2.41 | match: 3 |
| scalability/cvar_capped_24h_v1 | 1050624, 1050623, 1050628 | 300.01 | match: 11; adapter_error_unresolved_at_deadline: 2; timeout_or_unattempted: 3; mismatch: 1 |
| scalability/equal_screen_v1 | 1055139, 55135, 1055141 | 152.73 | timeout_or_unattempted: 1; match: 1; mismatch: 1 |
| scalability/msd_cvar_original | 50633, 50635, 50642 | 70.95 | match: 12; unavailable_configuration_or_solver: 6 |
| stochastic_fo/continuation_v2 | 3502125, 2502124, 3502124 | 297.05 | timeout_or_unattempted: 18 |
| stochastic_fo/continuation_v3 | 3502124, 2502125, 3502123 | 300.01 | timeout_or_unattempted: 18 |

Only two distinct game seeds were available among the missing-profile pilot rows;
that pilot's 84 matching comparisons are hyperparameter/method rows on those two
games, not 84 independent games. `timeout_or_unattempted` covers a method stopped
at the deadline or a later method not reached within its seed's shared budget.
Six older general FO rows lack a sufficiently recorded numerical configuration.
Two capped CVaR rows hit a sidecar-serialization adapter error before the budget
expired. The adapter was repaired, but those exhausted budgets were not extended.
The continuation-v2 constant-step adapter was also repaired; its remaining-budget
replays reached their deadlines without a completed method. Initial attempts and
repair attempts are retained separately, and repair time is charged to the same
seed and study budgets. Unresolved checks are inconclusive, not matches.

The three sampled uniform-game QPTAS replays agree. Other older campaigns contain
both matching and differing results; see the row-linked comparisons. Replayed
profiles are new outputs and never substitute for missing original profiles.
Even matching replays do not recertify every original result or estimate a new
success rate. Hardware and software differences also prevent a direct runtime
comparison with the recorded remote timings.

## Population v2 replay details

All selected games have n=50 and K=500. Each fresh replayed profile's own saved
regret agrees with a new full-sample LP certificate.

| Risk | Seed | Method | Original eta | Replayed eta | Comparison |
| --- | ---: | --- | ---: | ---: | --- |
| cvar | 6005140 | minibatch | 0.017937802 | 0.015125793 | mismatch |
| cvar | 6005140 | full_batch | 0.007512920 | 0.007426425 | mismatch |
| msd | 5005135 | full_batch | 0.017447598 | 0.017470630 | mismatch |
| msd | 5005135 | minibatch | 0.005776146 | 0.010655537 | classification_flip |
| cvar | 6005132 | minibatch | 0.016038742 | 0.011065760 | mismatch |
| cvar | 6005132 | full_batch | 0.009778275 | 0.009654205 | mismatch |

For MSD minibatch seed 5005135, the recorded eta was below 0.01 and the replayed
eta is above it. The original remote rows record NumPy 2.5.3; the local audit uses
NumPy 2.2.6. Full remote package versions, source revision, and input tensor hashes
are missing. The source of the differences is unresolved; these observations do
not prove that the original certificates were wrong, nor that NumPy alone caused
the differences. The local pre/post-refactor generator fixtures and FO checkpoint
capture tests preserve the same numerical path in the same environment.

## Evidence and commands

- [Original-profile checks](profile_checks.jsonl), [counts](summary.json), and
  [audit environment](environment.json).
- `row_inventory.json.gz`: losslessly compressed row inventory, with source paths,
  row numbers, original metrics, and stable comparison keys.
- [Frozen replay selection](replays/selection.json), [execution and attempt times](replays/execution.json),
  [row-linked comparison CSV](replays/comparisons.csv), and [replay summary](replays/summary.json).
- Per-seed replay directories retain tasks, comparisons, and newly saved profiles.
  Worker logs remain local under the repository's log-ignore policy.
- [Rounded notebook profile checks](rounded_notebook_profiles/profile_checks.jsonl).

Run from the repository root:

```bash
# Rebuild only the compact report; no solvers are run.
python -m experiments.diagnostics.replay \
  --audit-dir experiments/results/audits/consolidation_20260915 --summarize

# A new available-profile audit must use a new output directory.
python -m experiments.diagnostics.regret_audit \
  --output-dir experiments/results/audits/NEW_AUDIT
# Then freeze/select and execute its bounded missing-profile replays once.
python -m experiments.diagnostics.replay \
  --audit-dir experiments/results/audits/NEW_AUDIT
```

Do not rerun into this directory or overwrite frozen raw campaign results.
Future campaign runners save final strategies, certificates, checkpoint history,
input hashes, and software/source provenance outside the numerical timing region.
