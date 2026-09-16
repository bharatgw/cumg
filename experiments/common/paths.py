"""Catalog paths and the explicit historical relocation map."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPO = ROOT.parent


def catalog() -> dict:
    return json.loads((RESULTS / "catalog.json").read_text())


def result_path(campaign: str, artifact: str = "") -> Path:
    return REPO / catalog()["runs"][campaign]["path"] / artifact


def relocated_path(value: str) -> str:
    """Translate only known historical paths; never change numerical settings."""
    mapping = catalog()["relocations"]
    for old in sorted(mapping, key=len, reverse=True):
        if value == old or value.startswith(old + "/"):
            return mapping[old] + value[len(old) :]
    return value
