"""Load explicit campaign presets and validate historical resume configuration."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.common.paths import REPO, ROOT, catalog, relocated_path


def load_campaign(name: str, config: Path | None = None) -> dict:
    path = config or ROOT / "configs" / (name + ".json")
    if config is None and not path.resolve().is_relative_to(ROOT / "configs"):
        raise ValueError("Campaign must be a study/run ID")
    result = json.loads(path.read_text())
    if result["campaign"] != name:
        raise ValueError("Preset campaign ID differs from requested campaign")
    frozen_hash = catalog()["runs"].get(name, {}).get("preset_sha256")
    if frozen_hash and hashlib.sha256(path.read_bytes()).hexdigest() != frozen_hash:
        raise ValueError("Frozen campaign preset changed; use a new campaign ID")
    return result


def check_saved_config(path: Path, proposed: dict[str, str]) -> None:
    if not path.exists():
        return
    saved = dict(line.split("=", 1) for line in path.read_text().splitlines() if line)
    if saved.keys() != proposed.keys():
        raise ValueError("Configuration keys differ; use a new campaign ID")
    for key, value in saved.items():
        expected = proposed[key]
        if key == "LEGACY_RESULT_DIR":
            value, expected = relocated_path(value), relocated_path(expected)
        if value != expected:
            raise ValueError(f"Configuration differs at {key}; use a new campaign ID")


def resolver_args(config: dict) -> argparse.Namespace:
    env = config["historical_env"]
    result = REPO / config["result_dir"]
    return argparse.Namespace(
        result_dir=result,
        legacy_dir=REPO / relocated_path(env.get("LEGACY_RESULT_DIR", config["result_dir"])),
        risk=env["risk"].split(),
        K=[int(x) for x in env["K_GRID"].split()],
        n=[int(x) for x in env["N_GRID"].split()],
        reps=int(env["REPS"]),
        seed_base=int(env["SEED_BASE"]),
        methods=env["METHODS"].split(),
        time_limit_seconds=int(env["METHOD_TIME_LIMIT_SECONDS"]) if config["engine"] == "capped" else float("inf"),
        stochastic_n_random_starts=int(env.get("STOCHASTIC_N_RANDOM_STARTS", 0)),
        payoff_model=env.get("PAYOFF_MODEL", "uniform"),
        retry_errors=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--proposed", required=True)
    args = parser.parse_args()
    check_saved_config(args.config, dict(line.split("=", 1) for line in args.proposed.splitlines()))
