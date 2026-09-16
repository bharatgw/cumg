# Remote scalability and resume protocol

Use the [common campaign interface](README.md). Historical CVaR presets are
[`scalability/cvar_uncapped_v1`](configs/scalability/cvar_uncapped_v1.json) and
[`scalability/cvar_capped_24h_v1`](configs/scalability/cvar_capped_24h_v1.json).

```bash
python -m experiments plan --campaign scalability/cvar_capped_24h_v1
python -m experiments status --campaign scalability/cvar_capped_24h_v1
python -m experiments run --campaign scalability/cvar_capped_24h_v1
python -m experiments collect --campaign scalability/cvar_capped_24h_v1
```

The uncapped workflow writes a wide row after a replicate's configured methods
finish. The capped workflow resolves each method independently: it reuses
completed legacy measurements, censors measurements above 86,400 seconds, and
runs only unresolved tasks. A timeout or process failure gets a status JSON
marker; successful execution publishes a partial CSV atomically. A completed
unsuccessful search remains completed. Error retries preserve the old marker
and log; timeout records are not retried by that switch.

Stop old workers before migrating or resuming their output directories. Local
consolidation did not stop or relocate any remote process or file. Per-method
locks prevent duplicate work; do not remove a lock belonging to an active
worker. GNU `timeout` is required for capped execution, and PATH/PATHAMPL is
required for MCP methods. Pure QPTAS and FO jobs do not require external MCP
binaries.

Numerical settings cannot change within a saved campaign. The explicit catalog
relocation map permits only known historical path changes in saved configuration
comparisons. Historical raw configuration and environment files remain intact.
