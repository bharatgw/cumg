# Experiment consolidation implementation

Completed locally on 2026-09-15. The population-v2 README figure was added before
the refactor, as requested. The local plan remains ignored in
`experiments/scratch/experiments_consolidation_plan.md`.

## Changes

- One campaign CLI provides `plan`, `run`, `status`, `collect`, and `sync`.
  Named JSON presets preserve recorded settings and calibration phase arguments.
  Existing preset hashes and saved configurations reject incompatible resumes.
  New numerical configurations require a new run ID.
- Shared generators, seed logic, certificate-success reporting, option conversion,
  CSV writing, paths, and provenance live in `common/`. General comparisons and
  FO tuning retain separate execution loops in `runners/`. Existing capped and
  uncapped scheduling semantics, locks, error retry, and timeouts are retained.
- Notebooks, plotting, and loaders moved to `analysis/`; diagnostic tools moved
  to `diagnostics/`. Superseded entry points are removed from the active root;
  selected historical implementations remain in `legacy/` for provenance.
- Results use `<study>/<run>/`, with an explicit catalog and reversible migration
  manifest. All 7,996 historical research artifacts retain their bytes. Ignore
  rules preserve local-only material and keep formerly tracked logs eligible.
- New scheduled runs save versioned returned-profile sidecars, available
  certificates/best responses, CVaR thresholds, checkpoint diagnostics, effective
  parameters, input hashes, and source/software/environment provenance. Saving is
  outside the algorithm timing boundary. The only numerical-package change is
  optional retention of already computed FO certification checkpoints; it adds
  no LP solves, RNG draws, update changes, or stopping changes.
- README population plots use a separate dataset from uniform-game plots. Both
  count completed finite eta <= 0.01 and mark cells with fewer than five successes.
  Historical figure analysis values and existing scalability images are preserved.

## Verification

The automated suite passes: **321 tests passed, 15 external-solver tests excluded**.
The [validation record](results/audits/consolidation_20260915/implementation_validation.json)
also records matching pre-refactor figure summaries and migration verification.
It covers numerical solvers without requiring external binaries, generator and
sampling behavior, checkpoint capture, metadata round trips, campaign smoke runs,
resume/error retry, configuration guards, audit comparisons, timeout cleanup,
transfer failure propagation, and figure counts. External-solver-marked tests
were excluded locally; CI retains a separate solver job.

Additional acceptance checks:

- Seeded uniform/population arrays and population maps match pre-refactor fixtures
  bit-for-bit. FO checkpoint capture produces identical returned profiles, regrets,
  selected starts, and iteration counts for both risks.
- Migrated v2 resolves to 640 completed jobs, 135 recorded successes, and zero
  pending jobs. `run` launches nothing; `collect` preserves its original aggregate.
- The sampled uniform-game QPTAS preset still plans 960 jobs.
- Ruff lint/format, compilation, shell syntax, notebook source syntax, active
  reference checks, and figure regeneration pass.
- Migration hashes and old/new tracking eligibility are checked independently;
  no files were staged and the plan remains ignored.

## Scientific findings and limits

The [audit report](results/audits/consolidation_20260915/README.md) distinguishes
original-profile checks from bounded seed replays. All 3,333 comparable available
or fixed-profile checks agree with recorded regrets. Another 1,100 rounded
notebook profiles have fresh certificates but no original regret to compare.
Missing original payoff hashes make those regenerated-input checks conditional.

For population v2, all six sampled FO replay regrets differ from the remote
records, and one MSD minibatch result changes from success to failure at 0.01.
The original returned profiles and full remote environment are unavailable;
these are unresolved reproducibility findings, not proof of erroneous original
certificates. README and catalog notes preserve this qualification. Other older
studies include matching replays, mismatches, missing configurations, and checks
that could not finish within the five-minute study limit. No full campaign was
rerun and no historical scientific result was replaced.

No remote migration, synchronization, job cancellation, Git staging, or commit was
performed. Use the [experiment guide](README.md) for the new commands and explicit
old remote paths while the server still uses the historical layout.
