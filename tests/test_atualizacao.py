import numpy as np
import pandas as pd
import pytest

from triagem_baixo_risco.modeling.atualizacao import (
    estimate_prevalence_acc,
    prior_shift,
    quarter_starts,
    run_strategy,
    soft_weights,
    window_mask,
)
from triagem_baixo_risco.modeling.train import fit_window


def _series(n=24000, seed=0, shift_from=None, shift=0.0):
    """Probabilidades brutas calibradas, com datas diárias de 2019 a 2025 e deriva opcional de prevalência."""
    rng = np.random.default_rng(seed)
    dates = pd.Series(pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 7 * 365, n), unit="D"))
    p_raw = rng.beta(0.6, 9.0, n)
    logit = np.log(p_raw / (1 - p_raw))
    if shift_from is not None:
        logit = logit + np.where(dates >= pd.Timestamp(shift_from), shift, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return p_raw, y, dates


def test_quarter_starts_are_calendar_quarters():
    q = quarter_starts("2023-01-01", "2023-10-01")
    assert [d.strftime("%Y-%m-%d") for d in q] == ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]


def test_windows_respect_lag_and_exclude_training_rows():
    dates = pd.Series(pd.to_datetime(["2021-12-31", "2022-03-15", "2022-09-30", "2022-10-01", "2023-01-02"]))
    train = np.array([True, False, False, False, False])
    q = pd.Timestamp("2023-01-01")
    assert window_mask("b", q, dates, train).tolist() == [False, True, True, True, False]
    assert window_mask("c", q, dates, train).tolist() == [False, True, True, False, False]  # termina em 2022-10-01
    assert window_mask("f", q, dates, train).tolist() == [False, True, True, False, False]
    assert window_mask("d", q, dates, train).tolist() == window_mask("b", q, dates, train).tolist()


def test_soft_weights_halve_every_half_life():
    q = pd.Timestamp("2023-01-01")
    dates = pd.Series(pd.to_datetime(["2022-10-01", "2021-10-01", "2023-06-01"]))
    w = soft_weights(dates, q, lag=3, half_life=4.0)
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(0.5, abs=0.01) and w[2] == 0.0


def test_prevalence_estimate_tracks_a_label_shift():
    rng = np.random.default_rng(1)

    def draw(n, pi):
        y = (rng.random(n) < pi).astype(int)
        p = np.where(y == 1, rng.beta(4, 6, n), rng.beta(1, 12, n))
        return p, y

    p_c, y_c = draw(20000, 0.05)
    p_t, _ = draw(20000, 0.12)
    assert estimate_prevalence_acc(p_c, y_c, p_t, tau=0.2) == pytest.approx(0.12, abs=0.02)


def test_prior_shift_is_identity_without_shift_and_raises_risk_when_prevalence_rises():
    p = np.array([0.01, 0.1, 0.5])
    assert prior_shift(p, 0.05, 0.05) == pytest.approx(p)
    assert (prior_shift(p, 0.05, 0.10) > p).all()


def test_static_strategy_keeps_one_calibration_and_reports_every_quarter():
    p_raw, y, dates = _series()
    train = (dates < "2022-01-01").to_numpy()
    base = ((dates >= "2022-01-01") & (dates < "2023-01-01")).to_numpy()
    rows = run_strategy(
        "a",
        p_raw=p_raw,
        y=y,
        dates=dates,
        quarters=quarter_starts("2023-01-01", "2025-10-01"),
        base_mask=base,
        train_mask=train,
    )
    assert [r["trimestre"] for r in rows][:2] == ["2023T1", "2023T2"] and len(rows) == 12
    assert len({r["r_hat"] for r in rows}) == 1
    assert all(len(r["p"]) == len(r["index"]) == len(r["released"]) for r in rows)
    assert {r["n1"] for r in rows} == {fit_window(p_raw, y, base).rule.n1} and "grupos" not in rows[0]
    g = (p_raw > 0.05).astype(int)
    grouped = run_strategy(
        "a",
        p_raw=p_raw,
        y=y,
        dates=dates,
        quarters=quarter_starts("2023-01-01", "2023-04-01"),
        base_mask=base,
        train_mask=train,
        groups=g,
    )
    assert isinstance(grouped[0]["n1"], dict) and (grouped[0]["grupos"] == g[grouped[0]["index"]]).all()


def test_quarterly_recalibration_follows_a_prevalence_rise_that_breaks_the_static_rate_bound():
    p_raw, y, dates = _series(n=60000, shift_from="2024-01-01", shift=1.2)
    train = (dates < "2022-01-01").to_numpy()
    base = ((dates >= "2022-01-01") & (dates < "2023-01-01")).to_numpy()
    q = quarter_starts("2023-01-01", "2025-10-01")
    kw = dict(p_raw=p_raw, y=y, dates=dates, quarters=q, base_mask=base, train_mask=train)
    static, quarterly = run_strategy("a", **kw), run_strategy("c", **kw)

    def mean_oe(rows):
        return np.mean([y[r["index"]].mean() / r["p"].mean() for r in rows[-4:]])

    assert mean_oe(static) > 1.5
    assert abs(mean_oe(quarterly) - 1) < abs(mean_oe(static) - 1)


def test_aci_raises_alpha_when_the_rule_is_conservative():
    p_raw, y, dates = _series(n=40000, shift_from="2023-01-01", shift=-1.0)  # prevalência cai: regra fica conservadora
    train = (dates < "2022-01-01").to_numpy()
    base = ((dates >= "2022-01-01") & (dates < "2023-01-01")).to_numpy()
    rows = run_strategy(
        "e",
        p_raw=p_raw,
        y=y,
        dates=dates,
        quarters=quarter_starts("2023-01-01", "2025-10-01"),
        base_mask=base,
        train_mask=train,
        gamma=0.05,
    )
    assert rows[0]["alpha_usado"] == 0.05 and rows[1]["alpha_usado"] == 0.05
    assert rows[-1]["alpha_usado"] != 0.05


def test_retraining_strategy_calls_refit_once_per_year():
    p_raw, y, dates = _series()
    calls = []

    def refit(cutoff):
        calls.append(cutoff)
        return p_raw

    run_strategy(
        "d",
        p_raw=p_raw,
        y=y,
        dates=dates,
        quarters=quarter_starts("2023-01-01", "2024-10-01"),
        base_mask=((dates >= "2022-01-01") & (dates < "2023-01-01")).to_numpy(),
        train_mask=(dates < "2022-01-01").to_numpy(),
        refit=refit,
    )
    assert [c.year for c in calls] == [2022, 2023]


def test_window_with_too_few_deaths_reuses_the_last_calibration():
    p_raw, y, dates = _series(n=3000)
    train = (dates < "2019-10-01").to_numpy()
    base = ((dates >= "2019-10-01") & (dates < "2020-01-01")).to_numpy()
    rows = run_strategy(
        "c",
        p_raw=p_raw,
        y=y,
        dates=dates,
        quarters=quarter_starts("2020-01-01", "2020-01-01"),
        base_mask=base,
        train_mask=train,
    )
    assert rows[0]["fallback"] is True


def _label_shift_series(n=80000, seed=5, pi_before=0.05, pi_after=0.12, shift_from="2024-01-01"):
    """Deriva só de prevalência: P(escore | classe) fixa, pressuposto da estratégia g."""
    rng = np.random.default_rng(seed)
    dates = pd.Series(pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 7 * 365, n), unit="D"))
    pi = np.where(dates >= pd.Timestamp(shift_from), pi_after, pi_before)
    y = (rng.random(n) < pi).astype(int)
    p_raw = np.where(y == 1, rng.beta(4, 6, n), rng.beta(1, 12, n))
    return p_raw, y, dates


def test_prevalence_strategy_keeps_releases_and_raises_the_rate_bound_under_label_shift():
    p_raw, y, dates = _label_shift_series()
    kw = dict(
        p_raw=p_raw,
        y=y,
        dates=dates,
        quarters=quarter_starts("2024-01-01", "2024-10-01"),
        base_mask=((dates >= "2022-01-01") & (dates < "2023-01-01")).to_numpy(),
        train_mask=(dates < "2022-01-01").to_numpy(),
    )
    a, g = run_strategy("a", **kw), run_strategy("g", **kw)
    with pytest.raises(ValueError):
        run_strategy("g", groups=np.zeros(len(y), dtype=int), **kw)
    assert all((ra["released"] == rg["released"]).all() for ra, rg in zip(a, g))
    assert np.mean([rg["r_hat"] for rg in g]) > np.mean([ra["r_hat"] for ra in a])
    oe_a = np.mean([y[r["index"]].mean() / r["p"].mean() for r in a])
    oe_g = np.mean([y[r["index"]].mean() / r["p"].mean() for r in g])
    assert abs(oe_g - 1) < abs(oe_a - 1)
