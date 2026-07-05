import numpy as np

from cumgSolver.results import SolverConfig, SolverResult, SupportSearchConfig, SupportSearchResult


def test_solver_result_as_dict_keeps_extra_fields():
    result = SolverResult(
        x=np.array([1.0, 0.0]),
        y=np.array([0.25, 0.75]),
        alpha1=1.5,
        alpha2=-0.5,
        model="MSD",
        solver="pathampl",
        solve_time_s=0.1,
        extra={"status": "ok"},
    )

    out = result.as_dict()

    np.testing.assert_allclose(out["x"], result.x)
    np.testing.assert_allclose(out["y"], result.y)
    assert out["status"] == "ok"
    assert out["model"] == "MSD"
    assert out["solver"] == "pathampl"


def test_config_defaults_are_independent_dicts():
    solver_a = SolverConfig()
    solver_b = SolverConfig()
    search_a = SupportSearchConfig()
    search_b = SupportSearchConfig()

    solver_a.solver_options["time_limit"] = 1
    search_a.solver_options["major_iteration_limit"] = 20

    assert solver_b.solver_options == {}
    assert search_b.solver_options == {}


def test_support_search_result_defaults_to_unsolved_payload():
    result = SupportSearchResult(success=False, best_error="no candidates")

    assert result.x is None
    assert result.y is None
    assert result.support is None
    assert result.best_error == "no candidates"
