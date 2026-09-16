# Population QPTAS and FO campaign

The completed calibrated preset is
[`configs/population_qptas_fo/beta_uniform_v2.json`](configs/population_qptas_fo/beta_uniform_v2.json).

| Setting | Value |
| --- | --- |
| Risks | MSD and CVaR; gamma=0.5, CVaR alpha=0.5 |
| Grid | n=50; K=500, 1000, 2000, 4000; 20 seeds each |
| Methods | sampled QPTAS, screened QPTAS, FO full batch, FO minibatch |
| QPTAS budget | 1,000 candidates; kappa=8; screening tau=ceil(sqrt(K)) |
| Tolerance | epsilon=0.01; screen epsilon=2 epsilon/3 |
| FO budget | 2,000 updates per start; uniform plus up to four conditional random starts |
| FO certification | every 100 updates; return immediately at a passing certificate |
| FO stagnation | 500-update window; rtol=0.005, atol=1e-5 |
| FO regularization | entropy=0.01; smoothing tau=0.002 |
| FO steps | MSD 1000, CVaR 500; decay=0.5; logit bound=20; JIT enabled |
| Execution | 8 workers; 24-hour cap per method/replicate |

Every player's matrix cell receives a seed-dependent, uniformly sampled choice
of Beta(1,4), Beta(2,3), Beta(3,2), Beta(4,1), or Uniform[0,1]. Its K samples
come from that fixed population. Mapping and sample streams are separate;
for fixed seed/n, increasing K preserves the sample prefix. See the versioned
generator in [`common/experiment.py`](common/experiment.py).

```bash
python -m experiments status --campaign population_qptas_fo/beta_uniform_v2
python -m experiments.analysis.generate_readme_figures --study population
```

The 640 attempts completed, with 135 recorded FO successes and none for either
QPTAS variant. The sampled cap gives no nonexistence certificate. Screened
QPTAS rejected every pair before a full best-response solve.

Original final FO strategies and full environment provenance were not saved.
The bounded local replay found numerical differences in all six selected FO
results and one success-classification change; the local NumPy version differs
from the recorded remote version. Keep these descriptive recorded outcomes
separate from fresh-profile recertification. See the [audit evidence](results/audits/consolidation_20260915/)
and [campaign interfaces](README.md) for resume, sync, and provenance rules.
