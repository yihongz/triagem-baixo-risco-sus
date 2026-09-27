import numpy as np
import pandas as pd

from tests.dados_sinteticos import synthetic_raw
from triagem_baixo_risco.features import build_features
from triagem_baixo_risco.modeling.train import (
    CATEGORICAL,
    FEATURE_BLOCKS,
    NUMERIC,
    block_permutation_importance,
    fit_isotonic,
    fit_platt,
    fit_window,
    make_hgb,
    make_logistic_full,
    make_logistic_simple,
    model_frame,
    rule_based_release,
)


def _xy(n=3000, seed=0):
    raw = synthetic_raw(n=n, seed=seed)
    X, y = build_features(raw)
    return raw, model_frame(X), y["morte"].to_numpy()


def test_model_frame_types_months_as_text_categories():
    _, Xm, _ = _xy(200)
    assert list(Xm.columns) == NUMERIC + CATEGORICAL
    assert Xm["mes_internacao"].isin([str(m) for m in range(1, 13)]).all()
    assert Xm["idade_anos"].dtype == float


def test_all_three_models_fit_and_rank_better_than_chance():
    _, Xm, y = _xy(3000)
    tr, te = slice(0, 2000), slice(2000, 3000)
    from sklearn.metrics import roc_auc_score

    for pipe in (make_hgb(0), make_logistic_full(), make_logistic_simple()):
        pipe.fit(Xm.iloc[tr], y[tr])
        assert roc_auc_score(y[te], pipe.predict_proba(Xm.iloc[te])[:, 1]) > 0.65


def test_platt_and_isotonic_fix_a_known_miscalibration():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.01, 0.3, 20000)
    y = (rng.random(20000) < p_true).astype(int)
    p_bad = np.clip(p_true * 2.0, 0, 1)  # superestima o risco
    for fit in (fit_platt, fit_isotonic):
        cal = fit(p_bad[:10000], y[:10000])
        assert abs(cal(p_bad[10000:]).mean() - y[10000:].mean()) < 0.01


def test_calibrators_fall_back_to_identity_with_a_single_class():
    cal = fit_platt(np.array([0.1, 0.2]), np.array([0, 0]))
    assert cal(np.array([0.3])).tolist() == [0.3]


def test_rule_based_release_follows_the_three_conditions():
    strata = pd.DataFrame(
        {
            "idade": [30.0, 50.0, 50.0, 70.0],
            "urgencia": [True, False, True, True],
            "obstetrica": [False, False, False, True],
        }
    )
    assert rule_based_release(strata).tolist() == [True, True, False, True]


def test_block_permutation_importance_ranks_the_informative_block_first():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=4000), "b": rng.normal(size=4000)})
    y = (rng.random(4000) < 1 / (1 + np.exp(-3 * X["a"]))).astype(int)
    imp = block_permutation_importance(
        lambda d: 1 / (1 + np.exp(-3 * d["a"].to_numpy())), X, y, {"a": ["a"], "b": ["b"]}, n_repeats=3
    )
    assert imp.iloc[0]["bloco"] == "a" and imp.iloc[0]["queda_auroc"] > 0.2
    assert abs(imp.set_index("bloco").loc["b", "queda_auroc"]) < 1e-9


def test_feature_blocks_cover_every_model_column_exactly_once():
    cols = [c for block in FEATURE_BLOCKS.values() for c in block]
    assert sorted(cols) == sorted(NUMERIC + CATEGORICAL)


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


def test_fit_window_uses_disjoint_halves():
    p_raw, y, dates = _series(n=4000)
    f = fit_window(p_raw, y, np.ones(4000, dtype=bool))
    assert 1800 < len(f.p_B) < 2200
