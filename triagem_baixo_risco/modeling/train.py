"""Modelos de risco, calibradores de probabilidade e divisão da janela de calibração em calibrador e regra de
liberação."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from triagem_baixo_risco.modeling.predict import ReleaseRule, fit_release_rule

NUMERIC = ["idade_anos", "urgencia", "mesmo_municipio"]
CATEGORICAL = [
    "sexo",
    "carater",
    "especialidade",
    "cid_capitulo",
    "cid3",
    "proc_grupo",
    "proc_subgrupo",
    "munic_res",
    "munic_mov",
    "cnes",
    "mes_internacao",
    "dia_semana_internacao",
]
SIMPLE_NUMERIC = ["idade_anos", "urgencia"]
SIMPLE_CATEGORICAL = ["sexo", "cid_capitulo"]
FEATURE_BLOCKS = {
    "idade": ["idade_anos"],
    "sexo": ["sexo"],
    "carater": ["carater", "urgencia"],
    "especialidade": ["especialidade"],
    "cid": ["cid_capitulo", "cid3"],
    "procedimento": ["proc_grupo", "proc_subgrupo"],
    "local": ["munic_res", "munic_mov", "mesmo_municipio", "cnes"],
    "tempo": ["mes_internacao", "dia_semana_internacao"],
}


def _as_category_str(s: pd.Series) -> pd.Series:
    """Coluna como texto categórico: numéricas viram inteiro antes, e todo valor ausente vira "NA"."""
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce").astype("Int64")
    return s.astype(str).replace({"<NA>": "NA", "nan": "NA", "None": "NA", "": "NA"})


def model_frame(X: pd.DataFrame, numeric=NUMERIC, categorical=CATEGORICAL) -> pd.DataFrame:
    """Recorta e tipa as colunas do modelo: numéricas como float, categóricas como texto ("NA" para ausentes)."""
    out = pd.DataFrame(index=X.index)
    for c in numeric:
        out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in categorical:
        out[c] = _as_category_str(X[c])
    return out


def make_hgb(seed: int = 42, numeric=NUMERIC, categorical=CATEGORICAL) -> Pipeline:
    """Modelo principal: boosting (HistGradientBoostingClassifier) com as categóricas nativas do sklearn."""
    pre = ColumnTransformer(
        [
            ("num", "passthrough", list(numeric)),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    max_categories=250,
                ),
                list(categorical),
            ),
        ]
    )
    clf = HistGradientBoostingClassifier(
        categorical_features=[False] * len(numeric) + [True] * len(categorical),
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _logistic(numeric, categorical) -> Pipeline:
    """Pipeline comum às duas logísticas: idade por spline, as demais numéricas padronizadas, categóricas em
    one-hot."""
    others = [c for c in numeric if c != "idade_anos"]
    transformers = [
        (
            "idade",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("spl", SplineTransformer(n_knots=5, degree=3))]),
            ["idade_anos"],
        )
    ]
    if others:
        transformers.append(
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), others)
        )
    transformers.append(
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), list(categorical))
    )
    return Pipeline([("pre", ColumnTransformer(transformers)), ("clf", LogisticRegression(max_iter=5000, C=0.5))])


def make_logistic_full(numeric=NUMERIC, categorical=CATEGORICAL) -> Pipeline:
    """Linha de base: logística com as variáveis dadas (por padrão, todas), idade por spline, one-hot com
    frequência mínima 20."""
    return _logistic(numeric, categorical)


def make_logistic_simple() -> Pipeline:
    """Linha de base: logística com idade (spline), sexo, urgência e capítulo da CID."""
    return _logistic(SIMPLE_NUMERIC, SIMPLE_CATEGORICAL)


def rule_based_release(strata: pd.DataFrame) -> np.ndarray:
    """Linha de base sem modelo: libera se obstétrica, ou eletiva com menos de 60 anos, ou menos de 40 anos."""
    idade = strata["idade"].to_numpy(dtype=float)
    eletiva = ~strata["urgencia"].to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        return strata["obstetrica"].to_numpy(dtype=bool) | (eletiva & (idade < 60)) | (idade < 40)


def _logit(p) -> np.ndarray:
    """Logit com corte em [1e-6, 1 - 1e-6], para não estourar em mais ou menos infinito nas probabilidades
    extremas."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p, y, sample_weight=None) -> Callable[[np.ndarray], np.ndarray]:
    """Recalibração logística sobre o logit (Platt). Com uma classe só, devolve a identidade."""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return lambda q: np.asarray(q, dtype=float)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y, sample_weight=sample_weight)
    return lambda q: lr.predict_proba(_logit(q).reshape(-1, 1))[:, 1]


def fit_isotonic(p, y, sample_weight=None) -> Callable[[np.ndarray], np.ndarray]:
    """Recalibração isotônica. Com uma classe só, devolve a identidade."""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return lambda q: np.asarray(q, dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        np.asarray(p, dtype=float), y, sample_weight=sample_weight
    )
    return lambda q: np.clip(iso.predict(np.asarray(q, dtype=float)), 0.0, 1.0)


def fit_calibrator(kind: str, p, y, sample_weight=None) -> Callable[[np.ndarray], np.ndarray]:
    """Ajusta o calibrador pelo nome: "platt" ou "isotonic"."""
    return {"platt": fit_platt, "isotonic": fit_isotonic}[kind](p, y, sample_weight)


def block_permutation_importance(
    predict_proba: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y,
    blocks: dict,
    n_repeats: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Queda de AUROC ao permutar juntas as colunas de cada bloco (mesma permutação de linhas no bloco)."""
    y = np.asarray(y, dtype=int)
    base = roc_auc_score(y, predict_proba(X))
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in blocks.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:
                Xp[c] = X[c].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, predict_proba(Xp)))
        rows.append(
            {
                "bloco": name,
                "queda_auroc": float(np.mean(drops)),
                "ic_lo": float(np.quantile(drops, 0.025)),
                "ic_hi": float(np.quantile(drops, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("queda_auroc", ascending=False).reset_index(drop=True)


@dataclass
class Fitted:
    """Resultado de fit_window: o calibrador e a regra ajustados na janela, e a parte B usada no ajuste da regra."""

    calibrate: Callable[[np.ndarray], np.ndarray]
    rule: ReleaseRule
    prevalence: float  # prevalência (ponderada) na parte B da janela
    p_B: np.ndarray
    y_B: np.ndarray
    index_B: np.ndarray  # linhas da parte B


def fit_window(
    p_raw,
    y,
    mask,
    *,
    kind: str = "platt",
    alpha: float = 0.05,
    seed: int = 42,
    weights=None,
    groups=None,
    conf: float = 0.95,
) -> Fitted:
    """Divide a janela em A (calibrador) e B (regra), aleatoriamente e meio a meio, com a semente dada."""
    idx = np.flatnonzero(mask)
    in_a = np.random.default_rng(seed).random(len(idx)) < 0.5
    ia, ib = idx[in_a], idx[~in_a]
    wa = None if weights is None else weights[ia]
    wb = None if weights is None else weights[ib]
    cal = fit_calibrator(kind, p_raw[ia], y[ia], sample_weight=wa)
    p_b = cal(p_raw[ib])
    rule = fit_release_rule(p_b, y[ib], alpha, conf, None if groups is None else np.asarray(groups)[ib], wb)
    prev = float(np.average(y[ib], weights=wb)) if len(ib) else float("nan")
    return Fitted(cal, rule, prev, p_b, y[ib], ib)
