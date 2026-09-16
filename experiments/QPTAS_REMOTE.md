# Sampled QPTAS campaign

Use the common campaign interface with
[`qptas_scalability/sampled_1000_v1`](configs/qptas_scalability/sampled_1000_v1.json):

```bash
python -m experiments plan --campaign qptas_scalability/sampled_1000_v1
python -m experiments run --campaign qptas_scalability/sampled_1000_v1
python -m experiments status --campaign qptas_scalability/sampled_1000_v1
python -m experiments sync --campaign qptas_scalability/sampled_1000_v1 --remote USER@HOST --preview
```

The preset compares MSD and CVaR on independent Uniform[0,1] payoffs, with
K=5,10,30,100,250,500; n=5,10,20,50; 20 repetitions; gamma=0.5;
CVaR alpha=0.5; epsilon=0.01; up to 1,000 uniformly sampled distinct joint
kappa-uniform profiles, kappa=ceil(sqrt(n)). The cap is 24 hours per method.
Sampling exhaustion is a completed unsuccessful attempt, not an error or a
certificate of equilibrium nonexistence. Successful returns use full-sample
mixed best-response LP checks.

The new entry points replace the old QPTAS-specific shell and sync commands.
See [Experiments](README.md) for creating a new preset, error retries, remote
paths, and the distinction between recorded and recomputed certificates.
