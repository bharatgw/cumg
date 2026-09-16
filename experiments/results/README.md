# Experiment results

The canonical layout is `<study>/<run>/`. [catalog.json](catalog.json) identifies
analysis inputs, explicit configurations, predecessor runs, and audit coverage.
Raw historical CSVs, logs, configuration files, and metadata retain their original
bytes. Historical command strings inside these artifacts are provenance; use the
[new campaign commands](../README.md) to launch work.

## Current presentations

| Presentation | Sources |
| --- | --- |
| README uniform-game algorithms | `scalability/msd_cvar_original` (MSD), `scalability/cvar_capped_24h_v1` (CVaR), `qptas_scalability/sampled_1000_v1` |
| README uniform-profile baseline | `uniform_baseline/v1` |
| README heterogeneous-population figure | `population_qptas_fo/beta_uniform_v2` |
| Notebook screening comparison | `scalability/equal_screen_v1` |
| Notebook FO diagnostics | `stochastic_fo/continuation_v2` and `continuation_v3` |
| Calibration evidence | `population_fo_calibration/pilot_v1` and `cvar_steps_v2` |
| Early notebook tables | `archive/notebook_results` |

Both README comparisons count success only for completed attempts with a finite
recorded certificate at most 0.01. Crosses mark fewer than five successful seeds
per method/risk/n/K cell. Runtime includes unsuccessful attempts; capped CVaR
timeouts contribute 86,400 seconds. The uniform baseline measures certification
of the fixed profile, not an FO iteration-zero runtime. FO appears in the
separate population figure and remains omitted from the uniform-game figures.

Population v2 contains 640 completed attempts and 135 recorded successes. Neither
QPTAS variant succeeds; screened QPTAS rejects every pair before an LP. Final FO
strategies were not saved. Its three-seed local replay differs in all six FO
comparisons, with one change in success at 0.01. See the
[certificate and replay audit](audits/consolidation_20260915/README.md) before
using these recorded outcomes as independently reproduced evidence.

The audit reprices every available profile in the presentation inputs, including
failed profiles. It also includes the entire uniform baseline and saved calibration
profiles. Missing-profile checks use three frozen game seeds per study/run, small
games first, within five minutes total; unavailable configurations and timed-out
checks remain inconclusive. Original results are never overwritten by replays.

## Retained historical sources

Uncapped CVaR is retained in `scalability/cvar_uncapped_v1` as the capped campaign's
predecessor. `stochastic_fo/pilot_v1` through `pilot_v3` and `continuation_v1` retain
earlier diagnostics. Runtime estimation lives under `qptas_runtime/`; execution
smoke tests live under `smoke/`. Mixed or interrupted historical sources live
under `archive/`, including a deliberately ignored local-only CVaR snapshot.
The two empty `qptas_scalability/beta_uniform_v*` directories were mistaken sync
destinations, not completed population campaigns.

`pending_methods.tsv` is an invocation's initial worklist, not live completion.
Use `python -m experiments status --campaign STUDY/RUN` for current status.
The collector leaves an identical aggregate unchanged; collecting a differing
frozen aggregate requires a new output path for review.

## Migration integrity

[migration_20260915.json](migration_20260915.json) maps 7,996 historical research
artifacts to their new paths with SHA-256 hashes, sizes, and original tracking
status. All moved files passed verification. Previously tracked artifacts remain
eligible for tracking; 3,505 ignored local-only artifacts remain ignored. Nothing
was staged or committed by the consolidation. The plan and temporary validation
files remain ignored under `experiments/scratch/`.

```bash
python -m experiments.diagnostics.migration \
  --manifest experiments/results/migration_20260915.json
```

The migration tool preflights collisions and content changes before moving any
file. It supports `--apply` and `--reverse`; apply the same reviewed migration
separately to a stopped remote checkout before using its new paths. The local
migration did not modify the remote server. Until the remote is migrated, sync
using its explicit old source path as documented in [the experiment guide](../README.md).
