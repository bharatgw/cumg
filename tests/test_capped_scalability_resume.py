import argparse
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
SCRIPT = EXPERIMENTS / "capped_scalability_resume.py"
SPEC = importlib.util.spec_from_file_location("capped_scalability_resume", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capped_scalability_resume = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capped_scalability_resume
SPEC.loader.exec_module(capped_scalability_resume)


def grid_args(tmp_path, methods):
    return argparse.Namespace(
        legacy_dir=tmp_path / "legacy",
        result_dir=tmp_path / "capped",
        risk=["cvar"],
        K=[500],
        n=[10],
        reps=1,
        seed_base=123,
        methods=methods,
        time_limit_seconds=86_400,
        retry_errors=False,
        manifest=tmp_path / "capped" / "pending.tsv",
        output=tmp_path / "capped" / "collected.csv",
    )


def write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_legacy_results_are_reused_or_uniformly_reclassified_as_timeouts(tmp_path):
    args = grid_args(tmp_path, ["mcp", "screened_dual"])
    task = capped_scalability_resume.Task("cvar", 500, 10, 0, "mcp")
    legacy_path = capped_scalability_resume.legacy_result_path(args.legacy_dir, task)
    write_row(
        legacy_path,
        {
            "risk": "cvar",
            "K": 500,
            "n": 10,
            "seed": 6_001_123,
            "alpha": 0.5,
            "mcp_success": True,
            "mcp_time_s": 300,
            "mcp_eta": 0.001,
            "screened_dual_success": True,
            "screened_dual_time_s": 90_000,
            "screened_dual_eta": 0.009,
        },
    )

    mcp = capped_scalability_resume.resolve_task(task, args)
    screened = capped_scalability_resume.resolve_task(
        capped_scalability_resume.Task("cvar", 500, 10, 0, "screened_dual"),
        args,
    )

    assert mcp["status"] == "completed"
    assert mcp["success"] is True
    assert mcp["time_s"] == 300
    assert mcp["source"] == "legacy_reused"
    assert screened["status"] == "timeout"
    assert screened["success"] is False
    assert screened["censored"] is True
    assert screened["time_s"] == 86_400
    assert screened["observed_time_s"] == 90_000
    assert screened["uncapped_success"] is True
    assert screened["uncapped_eta"] == "0.009"

    counts = capped_scalability_resume.write_manifest(args)
    assert counts == {
        "total": 2,
        "scheduled": 0,
        "completed": 1,
        "timeout": 1,
    }
    assert args.manifest.read_text() == ""


def test_empty_legacy_shard_is_planned_with_original_absolute_seed(tmp_path):
    args = grid_args(tmp_path, ["action_dual"])
    task = capped_scalability_resume.Task("cvar", 500, 10, 0, "action_dual")
    legacy_path = capped_scalability_resume.legacy_result_path(args.legacy_dir, task)
    legacy_path.parent.mkdir(parents=True)
    legacy_path.touch()

    counts = capped_scalability_resume.write_manifest(args)

    assert counts == {
        "total": 1,
        "scheduled": 1,
        "pending": 1,
    }
    assert args.manifest.read_text() == "cvar\t500\t10\t0\taction_dual\t6001123\n"


def test_timeout_marker_is_collected_as_censored_result(tmp_path):
    args = grid_args(tmp_path, ["action_dual"])
    args.legacy_dir.mkdir(parents=True)
    record_args = argparse.Namespace(
        result_dir=args.result_dir,
        risk="cvar",
        K=500,
        n=10,
        rep=0,
        method="action_dual",
        seed_base=123,
        status="timeout",
        elapsed_s=86_405,
        time_limit_seconds=86_400,
        exit_code=124,
        message="cap reached",
    )
    capped_scalability_resume.record_status(record_args)

    record = capped_scalability_resume.resolve_task(
        capped_scalability_resume.Task("cvar", 500, 10, 0, "action_dual"),
        args,
    )
    counts = capped_scalability_resume.collect_results(args)

    assert record["status"] == "timeout"
    assert record["censored"] is True
    assert record["time_s"] == 86_400
    assert record["observed_time_s"] == 86_405
    assert record["source"] == "capped_run"
    assert counts == {"total": 1, "timeout": 1}
    with args.output.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "timeout"
    assert rows[0]["seed"] == "6001123"
    assert rows[0]["status_message"] == "cap reached"


def test_completed_capped_shard_takes_precedence_over_legacy(tmp_path):
    args = grid_args(tmp_path, ["restricted_mcp"])
    task = capped_scalability_resume.Task("cvar", 500, 10, 0, "restricted_mcp")
    write_row(
        capped_scalability_resume.legacy_result_path(args.legacy_dir, task),
        {
            "restricted_mcp_success": False,
            "restricted_mcp_time_s": 10,
            "restricted_mcp_eta": 0.02,
        },
    )
    write_row(
        capped_scalability_resume.method_shard_path(args.result_dir, task),
        {
            "restricted_mcp_success": True,
            "restricted_mcp_time_s": 12,
            "restricted_mcp_eta": 0.009,
        },
    )

    record = capped_scalability_resume.resolve_task(task, args)

    assert record["status"] == "completed"
    assert record["source"] == "capped_run"
    assert record["success"] is True
    assert record["eta"] == "0.009"
