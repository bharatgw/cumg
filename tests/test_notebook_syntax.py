from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "experiments" / "mcpAlgoAnalysis.ipynb",
    ROOT / "experiments" / "mcpSolvers.ipynb",
    ROOT / "notebooks" / "nonLinearMCPSolver.ipynb",
    ROOT / "notebooks" / "solver.ipynb",
)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_code_cells_use_python_310_syntax(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text())

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=str(notebook_path), feature_version=(3, 10))
        except SyntaxError as error:
            pytest.fail(
                f"{notebook_path.relative_to(ROOT)} cell {index} "
                f"({cell.get('id', 'no-id')}) is not valid Python 3.10: {error}"
            )
