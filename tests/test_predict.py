import numpy as np
import pytest
from scipy.stats import betabinom

from triagem_baixo_risco.modeling.predict import (
    apply_rule,
    binom_greater_pvalue,
    clopper_pearson_upper,
    conformal_quantile,
    death_rate_released,
    evaluate_guarantees,
    exact_ci,
    expected_cost,
    fdl_null,
    fit_release_rule,
    fraction_deaths_released,
    g1_aggregate_pvalue,
    g1_group_pvalue,
    g1_pvalue,
    holm,
    prediction_sets_class_conditional,
    weighted_conformal_quantile,
)


def _calibrated(n, rng, a=0.5, b=8.0):
    p = rng.beta(a, b, n)
    return p, (rng.random(n) < p).astype(int)


def test_weighted_quantile_matches_standard_quantile_with_equal_weights():
    rng = np.random.default_rng(0)
    for n in (5, 37, 500):
        s = rng.random(n)
        for alpha in (0.05, 0.1, 0.3):
            assert weighted_conformal_quantile(s, np.ones(n), alpha) == conformal_quantile(s, alpha)


def test_weighted_quantile_is_infinite_when_calibration_is_too_small():
    assert weighted_conformal_quantile(np.array([0.1, 0.2]), np.ones(2), alpha=0.05) == float("inf")


def test_weighted_quantile_fails_closed_when_every_weight_is_zero():
    assert weighted_conformal_quantile(np.array([0.1, 0.2, 0.3]), np.zeros(3), alpha=0.5) == float("inf")


def test_clopper_pearson_upper_matches_closed_form_for_zero_events():
    assert clopper_pearson_upper(0, 10, 0.95) == pytest.approx(1 - 0.05 ** (1 / 10))
    assert clopper_pearson_upper(5, 5) == 1.0


def test_exact_ci_contains_the_estimate():
    lo, hi = exact_ci(30, 1000)
    assert lo < 0.03 < hi


def test_binomial_test_detects_a_clear_excess_and_ignores_a_deficit():
    assert binom_greater_pvalue(20, 100, 0.05) < 0.001
    assert binom_greater_pvalue(2, 100, 0.05) > 0.5


def test_holm_adjusts_in_step_down_order():
    out = holm([0.01, 0.04, 0.03])
    assert out["p_adjusted"] == pytest.approx([0.03, 0.06, 0.06])
    assert out["reject"] == [True, False, False]


def test_rule_keeps_fraction_of_deaths_released_below_alpha_on_exchangeable_data():
    rng = np.random.default_rng(1)
    fdl = []
    for _ in range(200):
        p_c, y_c = _calibrated(3000, rng)
        p_t, y_t = _calibrated(3000, rng)
        rule = fit_release_rule(p_c, y_c, alpha=0.1)
        fdl.append(fraction_deaths_released(y_t, apply_rule(rule, p_t))["value"])
    assert np.mean(fdl) <= 0.1 + 0.005


def test_release_matches_exclusion_of_the_death_label_even_with_ties():
    rng = np.random.default_rng(2)
    p = rng.choice([0.01, 0.02, 0.05, 0.2, 0.6], 5000)
    y = (rng.random(5000) < p).astype(int)
    rule = fit_release_rule(p[:2500], y[:2500], alpha=0.1)
    released = apply_rule(rule, p[2500:])
    sets = prediction_sets_class_conditional(np.column_stack([1 - p[2500:], p[2500:]]), {0: 1.0, 1: rule.q1})
    assert (released == ~sets[:, 1]).all()


def test_cross_mondrian_protects_each_group_where_the_global_rule_fails():
    rng = np.random.default_rng(3)
    fdl_global, fdl_cross = {0: [], 1: []}, {0: [], 1: []}
    for _ in range(100):
        g = rng.integers(0, 2, 6000)
        p = np.where(g == 0, rng.beta(0.3, 30, 6000), rng.beta(2, 8, 6000))  # grupo 0: óbitos raros e de risco baixo
        y = (rng.random(6000) < p).astype(int)
        c, t = slice(0, 3000), slice(3000, 6000)
        glob = fit_release_rule(p[c], y[c], alpha=0.1)
        cross = fit_release_rule(p[c], y[c], alpha=0.1, groups_cal=g[c])
        for lv in (0, 1):
            m = g[t] == lv
            fdl_global[lv].append(fraction_deaths_released(y[t][m], apply_rule(glob, p[t][m]))["value"])
            fdl_cross[lv].append(fraction_deaths_released(y[t][m], apply_rule(cross, p[t][m], g[t][m]))["value"])
    assert np.nanmean(fdl_global[0]) > 0.2
    for lv in (0, 1):
        assert np.nanmean(fdl_cross[lv]) <= 0.1 + 0.02


def test_unseen_group_is_never_released():
    rule = fit_release_rule(
        np.array([0.01, 0.02, 0.5, 0.6] * 50), np.array([0, 0, 1, 1] * 50), alpha=0.1, groups_cal=np.array(["a"] * 200)
    )
    assert not apply_rule(rule, np.array([0.001]), groups=np.array(["b"])).any()


def test_weighted_rule_uses_effective_sample_size_for_the_rate_bound():
    rng = np.random.default_rng(4)
    p, y = _calibrated(4000, rng)
    plain = fit_release_rule(p, y, alpha=0.1)
    weighted = fit_release_rule(p, y, alpha=0.1, weights_cal=np.where(np.arange(4000) < 2000, 1.0, 0.01))
    assert weighted.n_released_cal < plain.n_released_cal
    assert weighted.r_hat >= plain.r_hat - 1e-3


def test_evaluate_guarantees_flags_a_violated_rate_bound():
    y = np.array([1] * 30 + [0] * 970)
    released = np.ones(1000, dtype=bool)
    out = evaluate_guarantees(y, released, alpha=0.05, r_hat=0.01, n1=600)
    assert out["g2"]["value"] == pytest.approx(0.03)
    assert out["g2"]["p_violation"] < 0.001
    assert out["g1"]["value"] == 1.0 and out["g1"]["p_violation"] < 0.001


def test_death_rate_released_handles_no_releases():
    assert np.isnan(death_rate_released(np.array([1, 0]), np.array([False, False]))["value"])


def test_fdl_null_is_the_beta_of_the_conformal_order_statistic():
    assert fdl_null(2640, 0.05) == (132.0, 2509.0)
    for n1 in (19, 20, 99, 100, 400, 1234, 2640):
        for alpha in (0.05, 0.10):
            a, b = fdl_null(n1, alpha)
            assert a + b == n1 + 1 and a == np.floor((n1 + 1) * alpha + 1e-9)
    assert fdl_null(100.5, 0.05) == pytest.approx((0.05 * 101.5, 0.95 * 101.5))
    assert fdl_null(0, 0.05) is None
    assert fdl_null(10, 0.05) is None  # k = 11 > n1: q1 infinito, ninguém é liberado


def test_g1_pvalue_is_the_beta_binomial_tail():
    null = fdl_null(400, 0.05)
    assert g1_pvalue(28, 400, null) == pytest.approx(betabinom.sf(27, 400, *null), rel=1e-12)
    assert g1_pvalue(0, 400, null) == 1.0 and g1_pvalue(3, 0, null) == 1.0
    rng = np.random.default_rng(5)
    x = rng.binomial(400, rng.beta(*null, 20000))  # modelo gerador: FDL condicional beta, depois binomial
    assert np.mean(x >= 28) == pytest.approx(g1_pvalue(28, 400, null), abs=0.01)


def test_g1_group_pvalue_convolves_independent_groups():
    nulls, ms, ks = [fdl_null(60, 0.1), fdl_null(45, 0.1)], [12, 9], [3, 2]
    pmf = [betabinom.pmf(np.arange(m + 1), m, *nl) for m, nl in zip(ms, nulls)]
    brute = sum(pmf[0][i] * pmf[1][j] for i in range(13) for j in range(10) if i + j >= 5)
    assert g1_group_pvalue(ks, ms, nulls) == pytest.approx(brute, rel=1e-9)
    assert g1_group_pvalue([0, 0], ms, nulls) == 1.0


def test_g1_aggregate_is_exact_with_one_calibration_and_above_the_independent_case_across_calibrations():
    null = fdl_null(500, 0.05)
    assert abs(g1_aggregate_pvalue(40, [200, 300, 150], [null] * 3) - g1_pvalue(40, 650, null)) < 1e-12
    nulls, ms = [fdl_null(200, 0.05), fdl_null(300, 0.05)], [100, 150]
    pmf = [betabinom.pmf(np.arange(m + 1), m, *nl) for m, nl in zip(ms, nulls)]
    independent = np.convolve(pmf[0], pmf[1])[20:].sum()  # k = 20, acima da média (cerca de 12,5)
    p = g1_aggregate_pvalue(20, ms, nulls)
    assert 0.0 <= p <= 1.0 and p >= independent


def test_fit_release_rule_records_the_deaths_behind_q1():
    rng = np.random.default_rng(6)
    p, y = _calibrated(4000, rng)
    g = rng.integers(0, 3, 4000)
    assert fit_release_rule(p, y, alpha=0.1).n1 == int(y.sum())
    assert fit_release_rule(p, y, alpha=0.1, groups_cal=g).n1 == {lv: int(y[g == lv].sum()) for lv in range(3)}
    w = rng.uniform(0.1, 1.0, 4000)
    wd = w[y == 1]
    assert fit_release_rule(p, y, alpha=0.1, weights_cal=w).n1 == pytest.approx(wd.sum() ** 2 / (wd**2).sum())
    by_group = fit_release_rule(p, y, alpha=0.1, groups_cal=g, weights_cal=w).n1
    wg = w[(y == 1) & (g == 2)]
    assert by_group[2] == pytest.approx(wg.sum() ** 2 / (wg**2).sum())
    assert fit_release_rule(p, np.zeros(4000, dtype=int), alpha=0.1).n1 == 0


def test_beta_binomial_g1_test_keeps_its_level_where_the_binomial_test_does_not():
    rng = np.random.default_rng(2026)
    reject_bb, reject_binom, reps = 0, 0, 2000
    for _ in range(reps):
        rule = fit_release_rule(rng.beta(2, 8, 400), np.ones(400, dtype=int), alpha=0.05)  # só óbitos definem q1
        k = int(apply_rule(rule, rng.beta(2, 8, 400)).sum())
        reject_bb += g1_pvalue(k, 400, fdl_null(rule.n1, 0.05)) < 0.05
        reject_binom += binom_greater_pvalue(k, 400, 0.05) < 0.05
    assert reject_bb / reps <= 0.065
    assert reject_binom / reps > 0.065  # o teste binomial ignora a variação da calibração


def test_evaluate_guarantees_uses_the_group_nulls_of_a_cross_rule():
    y = np.array([1] * 30 + [1] * 20 + [0] * 50)
    g = np.array(["a"] * 30 + ["b"] * 20 + ["a"] * 50)
    released = np.zeros(100, dtype=bool)
    released[[0, 1, 30]] = True
    n1 = {"a": 300, "b": 250}
    out = evaluate_guarantees(y, released, 0.05, 0.01, n1=n1, groups=g)["g1"]
    assert out["p_violation"] == pytest.approx(
        g1_group_pvalue([2, 1], [30, 20], [fdl_null(300, 0.05), fdl_null(250, 0.05)])
    )
    assert out["null"] is None
    one = evaluate_guarantees(y[:30], released[:30], 0.05, 0.01, n1=n1, groups=g[:30])["g1"]
    assert one["null"] == list(fdl_null(300, 0.05))
    assert one["p_violation"] == pytest.approx(g1_pvalue(2, 30, fdl_null(300, 0.05)), rel=1e-9)
    with pytest.raises(ValueError):
        evaluate_guarantees(y, released, 0.05, 0.01, n1=n1)


def test_conformal_quantile_applies_finite_sample_correction():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # n=5: k = ceil((n+1)(1-alpha)); alpha=0.2 -> ceil(4.8)=5 -> 0.5; alpha=0.5 -> ceil(3.0)=3 -> 0.3
    assert conformal_quantile(scores, alpha=0.2) == 0.5
    assert conformal_quantile(scores, alpha=0.5) == 0.3


def test_class_conditional_sets_use_per_label_thresholds():
    from triagem_baixo_risco.modeling.predict import prediction_sets_class_conditional

    probs = np.array([[0.7, 0.3]])
    sets = prediction_sets_class_conditional(probs, {0: 0.2, 1: 0.8})
    assert sets.tolist() == [[False, True]]


def test_expected_cost_averages_fn_fp_and_deferral_costs():
    y = np.array([1, 0, 1, 0])
    decision = np.array([0, 1, -1, 0])  # FN, FP, encaminhado, correto
    assert expected_cost(y, decision, c_fn=10, c_fp=1, c_defer=2) == pytest.approx((10 + 1 + 2) / 4)


def test_decide_with_band_uses_two_thresholds():
    from triagem_baixo_risco.modeling.predict import decide_with_band

    p1 = np.array([0.95, 0.3, 0.02])
    assert decide_with_band(p1, t_lo=0.1, t_hi=0.6).tolist() == [1, -1, 0]


def test_choose_band_collapses_to_cost_optimal_threshold_when_deferral_is_too_expensive():
    from triagem_baixo_risco.modeling.predict import choose_band

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)  # calibrado por construção
    t_lo, t_hi = choose_band(y, p, grid=np.round(np.arange(0.02, 1.0, 0.02), 2), c_fn=10, c_fp=1, c_defer=100)
    assert t_lo == t_hi
    assert 0.04 <= t_hi <= 0.16  # ótimo teórico: c_fp / (c_fn + c_fp) = 0.09
