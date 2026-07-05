from cumgSolver import available_solvers, format_solver_availability, solver_available


def test_solver_available_reports_boolean_for_unknown_solver():
    assert solver_available("definitely-not-a-real-cumg-solver") is False


def test_available_solvers_preserves_requested_names():
    statuses = available_solvers(["definitely-not-a-real-cumg-solver", "also-missing"])

    assert statuses == {
        "definitely-not-a-real-cumg-solver": False,
        "also-missing": False,
    }


def test_format_solver_availability_is_human_readable():
    summary = format_solver_availability(["definitely-not-a-real-cumg-solver"])

    assert summary == "definitely-not-a-real-cumg-solver=missing"
