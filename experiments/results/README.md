# Experiment result provenance

This directory contains curated outputs and raw shards from several distinct
campaigns. Similar directory names do not imply interchangeable specifications.

## Inputs used by `mcpAlgoAnalysis.ipynb`

- `remote/msd_cvar_part_scalability/`: original wide scalability shards. The
  notebook uses only its MSD rows because CVaR was rerun under a time cap.
- `remote/cvar_scalability/capped_24h_v1/capped_method_results.csv`: authoritative
  CVaR analysis table. It contains one row per method and preserves completed
  versus 24-hour right-censored outcomes.
- `remote/higher_eps/v1/`: equal screening-threshold comparison with
  `epsilon_scr=0.01` for MSD and CVaR. It contains 240 non-empty result shards.
- `uniform/`: uniform-profile baseline and its aggregate summary.
- `stochastic/continuation_shards/v2/` and `v3/`: stochastic continuation pilot
  summaries and histories used by the notebook's tuning diagnostics.

## Inputs used by the README figures

`generate_readme_figures.py` uses the same MSD and capped CVaR sources for MCP,
screened dual, action dual, and restricted MCP, plus
`remote/qptas_scalability/sampled_1000_v1/capped_method_results.csv` for sampled
QPTAS. Both figures also include `uniform/uniform_profile_baseline.csv`, matched
to the same 20 seeds in each displayed cell. Its `uniform_time_s` column measures
full-regret certification of the fixed uniform profile, excluding game generation.
It is a separate baseline measurement, not a recovered FO checkpoint time.

Stochastic full-batch and minibatch results are currently omitted from the README
figures. Their original data and notebook analysis are retained. Both figures
use finite certificates at most 0.01 to count successes, and cross markers for
cells with fewer than five successful seeds.

## Supporting and legacy outputs

- `remote/cvar_scalability/v1/`: uncapped source campaign used by the capped
  resume process. Keep it as provenance; do not use it as the final CVaR table.
- `remote/cvar_scalability/smoke/` and `remote/smoke/`: execution smoke tests.
- `stochastic/continuation_shards/v1/` and `stochastic/pilot/`: earlier pilot
  versions retained for comparison.
- `legacy/`: early aggregate tables from the original notebooks and scripts.
- `higher_eps/v1/`: legacy mixed-configuration run. Its MSD rows use
  `epsilon_scr=0.01`, while its CVaR rows use the historical default
  `epsilon_scr=2*epsilon/3`. The directory is incomplete and is not an input to
  the current notebook. `higher_eps/v1/missing_runs.csv` records the removed
  zero-byte placeholders.

`pending_methods.tsv` in a capped campaign is the worklist generated at the
start of that invocation. It is not a live or final status table. Use
`capped_method_results.csv` for the collected final state and method shards,
status markers, and logs while a campaign is running.

For new campaigns, write temporary shards to an ignored directory first and
promote only a documented aggregate, configuration, environment record, and
scientifically relevant failure logs into this tree.
