import numpy as np
import pytest

from triagem_baixo_risco.avaliacao import evaluate_period, h1, h2, h3, h4, h5
from triagem_baixo_risco.modeling.predict import binom_greater_pvalue, fdl_null, g1_pvalue


def test_observed_expected_and_intercept_slope_on_calibrated_data():
    from triagem_baixo_risco.avaliacao import calibration_intercept_slope, observed_expected

    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.4, 40000)
    y = (rng.random(40000) < p).astype(int)
    assert observed_expected(y, p) == pytest.approx(1.0, abs=0.03)
    intercept, slope = calibration_intercept_slope(y, p)
    assert abs(intercept) < 0.05 and slope == pytest.approx(1.0, abs=0.08)


def test_intercept_detects_overprediction():
    from triagem_baixo_risco.avaliacao import calibration_intercept_slope, observed_expected

    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.2, 40000)
    y = (rng.random(40000) < p).astype(int)
    p_high = np.clip(p * 1.5, 0, 1)
    assert observed_expected(y, p_high) < 0.75
    assert calibration_intercept_slope(y, p_high)[0] < -0.3


def test_ici_is_small_when_calibrated_and_large_when_not():
    from triagem_baixo_risco.avaliacao import integrated_calibration_index

    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.5, 30000)
    y = (rng.random(30000) < p).astype(int)
    assert integrated_calibration_index(y, p) < 0.01
    assert integrated_calibration_index(y, np.clip(p * 1.6, 0, 1)) > 0.05


def test_bootstrap_ci_covers_the_mean_and_cluster_version_is_wider():
    from triagem_baixo_risco.avaliacao import bootstrap_ci

    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(20), 200)
    x = rng.normal(size=4000) + np.repeat(rng.normal(scale=1.0, size=20), 200)
    lo, hi = bootstrap_ci(lambda i: x[i].mean(), len(x), n_boot=300)
    clo, chi = bootstrap_ci(lambda i: x[i].mean(), len(x), n_boot=300, clusters=clusters)
    assert lo < x.mean() < hi
    assert (chi - clo) > (hi - lo)


def test_calibration_curve_points_follow_the_diagonal_when_calibrated():
    from triagem_baixo_risco.avaliacao import calibration_curve_points

    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.6, 30000)
    y = (rng.random(30000) < p).astype(int)
    c = calibration_curve_points(y, p)
    assert np.max(np.abs(np.array(c["p"]) - np.array(c["observado_suavizado"]))) < 0.05
    assert sum(c["histograma_obitos"]) + sum(c["histograma_sobreviventes"]) == 30000


def test_analytic_intervals_cover_the_truth_on_calibrated_data():
    from triagem_baixo_risco.avaliacao import auroc_ci, calibration_intercept_slope_ci, oe_ci

    rng = np.random.default_rng(5)
    p = rng.uniform(0.01, 0.4, 30000)
    y = (rng.random(30000) < p).astype(int)
    lo, hi = oe_ci(y, p)
    assert lo < 1.0 < hi and hi - lo < 0.1
    c = calibration_intercept_slope_ci(y, p)
    assert c["intercepto_ic"][0] < 0 < c["intercepto_ic"][1] and c["slope_ic"][0] < 1 < c["slope_ic"][1]
    from sklearn.metrics import roc_auc_score

    a_lo, a_hi = auroc_ci(y, p)
    assert a_lo < roc_auc_score(y, p) < a_hi


def test_calibration_band_brackets_the_curve():
    from triagem_baixo_risco.avaliacao import calibration_curve_points

    rng = np.random.default_rng(6)
    p = rng.uniform(0.01, 0.5, 8000)
    y = (rng.random(8000) < p).astype(int)
    c = calibration_curve_points(y, p, n_points=20, n_boot=30)
    assert all(
        lo <= mid <= hi for lo, mid, hi in zip(c["banda_inferior"], c["observado_suavizado"], c["banda_superior"])
    )


def test_calibration_metrics_return_nan_for_a_single_class_so_cluster_bootstrap_survives():
    from triagem_baixo_risco.avaliacao import (
        bootstrap_ci,
        calibration_intercept_slope,
        calibration_intercept_slope_ci,
        integrated_calibration_index,
        smoothed_calibration,
    )

    y0, p0 = np.zeros(100, dtype=int), np.full(100, 0.05)
    assert all(np.isnan(v) for v in calibration_intercept_slope(y0, p0))
    ci = calibration_intercept_slope_ci(y0, p0)
    assert np.isnan(ci["slope"]) and all(np.isnan(v) for v in ci["slope_ic"] + ci["intercepto_ic"])
    assert np.isnan(integrated_calibration_index(y0, p0))
    assert np.isnan(smoothed_calibration(y0, p0)(np.array([0.1, 0.2]))).all()
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(10), 200)
    p = rng.uniform(0.01, 0.3, 2000)
    y = ((rng.random(2000) < p) & (clusters < 3)).astype(int)  # óbitos só em 3 dos 10 hospitais
    lo, hi = bootstrap_ci(lambda i: calibration_intercept_slope(y[i], p[i])[1], 2000, n_boot=60, clusters=clusters)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


NULL = fdl_null(2000, 0.05)  # nulo de G1 de uma regra com 2.000 óbitos na parte B


def _row(fdl_k, fdl_n, tol_k, tol_n, r_hat, ratio, p_g1=None, p_g2=None, oe=1.0, t="q"):
    return {
        "trimestre": t,
        "fdl_k": fdl_k,
        "fdl_n": fdl_n,
        "fdl": fdl_k / fdl_n,
        "tol_k": tol_k,
        "tol_n": tol_n,
        "tol": tol_k / tol_n,
        "r_hat": r_hat,
        "razao_prevalencia": ratio,
        "oe": oe,
        "g1_nulo": list(NULL),
        "p_g1": g1_pvalue(fdl_k, fdl_n, NULL) if p_g1 is None else p_g1,
        "p_g2": binom_greater_pvalue(tol_k, tol_n, r_hat) if p_g2 is None else p_g2,
    }


def test_evaluate_period_reports_both_guarantees_and_calibration():
    rng = np.random.default_rng(0)
    p = rng.beta(0.6, 9, 5000)
    y = (rng.random(5000) < p).astype(int)
    m = evaluate_period(y, p, p < 0.02, alpha=0.05, r_hat=0.01, prevalence_cal=y.mean(), n1=400)
    assert m["razao_prevalencia"] == pytest.approx(1.0) and m["g1_nulo"] == list(fdl_null(400, 0.05))
    assert 0 < m["fracao_liberada"] < 1 and m["oe"] == pytest.approx(1.0, abs=0.1)
    assert {"p_g1", "p_g2", "auroc", "slope"} <= set(m)
    assert m["oe_ic"][0] < m["oe"] < m["oe_ic"][1] and m["auroc_ic"][0] < m["auroc"] < m["auroc_ic"][1]


def test_bootstrap_metrics_reports_row_and_hospital_intervals():
    from triagem_baixo_risco.avaliacao import bootstrap_metrics

    rng = np.random.default_rng(1)
    p = rng.beta(0.6, 9, 4000)
    y = (rng.random(4000) < p).astype(int)
    out = bootstrap_metrics(y, p, p < 0.02, clusters=np.repeat(np.arange(20), 200), n_boot=50)
    assert set(out) == {"auroc", "oe", "fdl", "tol", "fracao_liberada"}
    assert all(len(v["linhas"]) == 2 and len(v["hospitais"]) == 2 for v in out.values())


def test_h1_holds_when_every_period_respects_alpha_and_fails_on_a_systematic_excess():
    ok = {"x": [_row(4, 100, 1, 500, 0.01, 1.0) for _ in range(8)]}
    assert h1(ok, 0.05)["refutada"] is False
    creeping = {"x": [_row(8, 100, 1, 500, 0.01, 1.0) for _ in range(12)]}  # 8% em cada trimestre, nenhum isolado forte
    out = h1(creeping, 0.05)
    assert out["refutada"] is True and out["cenarios"]["x"]["agregado"]["valor"] == pytest.approx(0.08)


def test_h2_supported_when_rate_breaks_only_under_higher_prevalence():
    rows = [_row(3, 100, 2, 1000, 0.004, r) for r in (0.8, 0.9, 1.0)] + [
        _row(3, 100, 20, 1000, 0.004, r) for r in (1.3, 1.5)
    ]
    assert h2({"x": rows})["refutada"] is False


def test_h2_refuted_when_rate_breaks_under_lower_prevalence():
    rows = [_row(3, 100, 20, 1000, 0.004, 0.8)] + [_row(3, 100, 20, 1000, 0.004, r) for r in (1.3, 1.5)]
    assert h2({"x": rows})["refutada"] is True


def test_h3_supported_when_quarterly_recalibration_is_closer_to_one():
    pairs = [({"oe": 1.3 + 0.01 * i, "p_g2": 0.001}, {"oe": 1.02, "p_g2": 0.5}) for i in range(10)]
    out = h3(pairs)
    assert out["refutada"] is False and out["violacoes_estatica"] == 10


def test_h3_refuted_when_there_is_no_difference():
    pairs = [({"oe": 1.0 + 0.01 * i, "p_g2": 0.5}, {"oe": 1.0 + 0.01 * i, "p_g2": 0.5}) for i in range(10)]
    assert h3(pairs)["refutada"] is True


def test_h4_checks_fractions_and_baseline_intervals():
    assert h4(0.62, 0.55, {"regra": (0.02, 0.05), "simples": (0.01, 0.03)})["refutada"] is False
    out = h4(0.62, 0.35, {"regra": (-0.01, 0.05)})
    assert out["refutada"] is True and out["falhas"]["fracao_adultos"] and out["falhas"]["vs_regra"]


def test_h5_marks_small_groups_as_descriptive():
    groups = {"0-17": [_row(9, 40, 1, 100, 0.01, 1.0)], "60-79": [_row(40, 1000, 1, 100, 0.01, 1.0)]}
    out = h5(groups, 0.05, min_deaths=100)
    assert out["faixas"]["0-17"]["descritiva"] is True
    assert out["refutada"] is False


def test_surge_codes_flag_diagnoses_far_above_their_reference_share():
    from triagem_baixo_risco.avaliacao import surge_codes

    ref = ["I21"] * 90 + ["A90"] * 1 + ["J18"] * 9
    period = ["I21"] * 60 + ["A90"] * 30 + ["J18"] * 8 + ["U07"] * 12
    assert surge_codes(ref, period) == {"A90", "U07"}


def test_residual_g1_removes_only_surge_deaths():
    import pandas as pd

    from triagem_baixo_risco.avaliacao import residual_g1

    dates = pd.Series(pd.to_datetime(["2023-03-01"] * 100 + ["2024-01-15"] * 60))
    y = np.array([1] * 100 + [1] * 60)
    codes = np.array(["I21"] * 100 + ["I21"] * 30 + ["A90"] * 30)
    index = np.arange(100, 160)
    released = np.array([False] * 30 + [True] * 30)  # só os óbitos por A90 foram liberados
    out = residual_g1(y, released, codes, dates, pd.Timestamp("2024-01-01"), index, alpha=0.05, n1=400)
    assert out["cids_em_surto"] == ["A90"] and out["fdl_sem_surto"] == 0.0 and out["fdl_sem_surto_n"] == 30
    assert out["p_g1_sem_surto"] == 1.0
    with pytest.raises(ValueError):
        residual_g1(y, released, codes, dates, pd.Timestamp("2024-01-01"), index, alpha=0.05, n1={"a": 400})


def test_h1b_uses_the_residual_counts():
    from triagem_baixo_risco.avaliacao import h1b

    rows = [
        dict(_row(12, 100, 1, 500, 0.01, 1.0), fdl_sem_surto_k=4, fdl_sem_surto_n=90, p_g1_sem_surto=0.6)
        for _ in range(4)
    ]
    assert h1({"x": rows}, 0.05)["refutada"] is True
    assert h1b({"x": rows}, 0.05)["refutada"] is False


def test_evaluate_period_survives_a_period_without_deaths_or_releases():
    y = np.zeros(200, dtype=int)
    p = np.full(200, 0.3)
    m = evaluate_period(y, p, np.zeros(200, dtype=bool), alpha=0.05, r_hat=0.01, prevalence_cal=0.05, n1=400)
    assert m["eventos"] == 0 and np.isnan(m["auroc"]) and np.isnan(m["fdl"]) and np.isnan(m["tol"])
    assert m["p_g1"] == 1.0 and m["p_g2"] == 1.0


def test_h5_is_refuted_only_by_holm_across_bands_and_periods():
    k_sig = next(k for k in range(200) if g1_pvalue(k, 200, NULL) < 0.05)  # FDL da faixa significativa sozinha
    bands = ["0-17", "18-39", "40-59", "60-79"]

    def groups(p_first):
        rows = [(p_first, k_sig), (0.57, 9), (0.57, 9), (0.65, 8)]
        return {
            b: [{"trimestre": "q", "p_g1": p, "fdl_k": k, "fdl_n": 200, "g1_nulo": list(NULL)}]
            for b, (p, k) in zip(bands, rows)
        }

    weak = h5(groups(0.047), 0.05, min_deaths=100)
    assert weak["refutada"] is False
    assert weak["faixas"]["0-17"]["agregado_descritivo"]["p_violacao"] < 0.05  # o agregado não refuta mais
    assert h5(groups(0.001), 0.05, min_deaths=100)["refutada"] is True


def test_h2_spearman_ignores_periods_without_a_defined_rate():
    rows = [_row(3, 100, 2, 1000, 0.004, 0.9), _row(3, 100, 4, 1000, 0.004, 1.05), _row(3, 100, 6, 1000, 0.004, 1.1)]
    no_release = dict(_row(3, 100, 0, 1, 0.004, 0.95), tol_k=0, tol_n=0, tol=float("nan"), p_g2=1.0)
    out = h2({"x": rows[:1] + [no_release] + rows[1:]})["cenarios"]["x"]
    assert out["spearman"] == pytest.approx(1.0) and out["criterios"] == [False, False, False]


def test_g1_aggregate_needs_the_beta_parameters_of_every_row_with_released_deaths():
    ok = _row(4, 100, 1, 500, 0.01, 1.0)
    with pytest.raises(KeyError):  # resultado antigo, sem g1_nulo
        h1({"x": [ok, {k: v for k, v in ok.items() if k != "g1_nulo"}]}, 0.05)
    with pytest.raises(ValueError, match="agregado de G1 indefinido"):
        h1({"x": [ok, dict(ok, g1_nulo=None)]}, 0.05)
    silent = dict(_row(0, 100, 1, 500, 0.01, 1.0), g1_nulo=None)  # regra que não libera óbitos
    agg = h1({"x": [ok, silent]}, 0.05)["cenarios"]["x"]["agregado"]
    assert agg["k"] == 4 and agg["p_violacao"] == pytest.approx(g1_pvalue(4, 100, NULL), abs=1e-5)
