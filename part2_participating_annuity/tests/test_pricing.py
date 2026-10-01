from src.pricing import Params, analytical_value, binomial_tree_value, monte_carlo_value

P = Params()


def test_analytical_matches_report():
    gmab, gmdb, total = analytical_value(P)
    assert abs(gmab - 98498.797) < 0.01
    assert abs(gmdb - 10102.893) < 0.05
    assert abs(total - 108601.690) < 0.05


def test_binomial_tree_converges_to_analytical():
    ref = analytical_value(P)[2]
    assert abs(binomial_tree_value(P, N=2000) - ref) < 5.0


def test_monte_carlo_within_statistical_error():
    ref = analytical_value(P)[2]
    est = monte_carlo_value(P, N=250, M=20_000, seed=0)
    assert abs(est - ref) < 1000  # a few standard errors
