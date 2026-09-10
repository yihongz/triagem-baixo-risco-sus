"""Métricas de calibração e discriminação, avaliação por período e critérios de refutação de H1 a H5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from triagem_baixo_risco.dataset import in_window
from triagem_baixo_risco.modeling.predict import (
    binom_greater_pvalue,
    evaluate_guarantees,
    exact_ci,
    fdl_null,
    g1_aggregate_pvalue,
    g1_pvalue,
    holm,
)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def observed_expected(y, p) -> float:
    """Razão O/E: eventos observados sobre esperados (calibração na média)."""
    return float(np.mean(y) / np.mean(p))


def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Intercepto e slope de calibração (Van Calster et al., 2019); o intercepto é estimado com o logit de p como offset."""
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")  # uma classe só (ex.: réplica de bootstrap sem óbitos)
    lp = _logit(p)
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y.astype(int)).coef_[0, 0])
    nll = lambda a: -np.mean(y * (a + lp) - np.logaddexp(0.0, a + lp))
    intercept = float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)
    return intercept, slope


def calibration_intercept_slope_ci(y, p, conf: float = 0.95) -> dict:
    """Intercepto (com offset) e slope com IC de Wald pela informação de Fisher."""
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        nan2 = (float("nan"), float("nan"))
        return {"intercepto": float("nan"), "intercepto_ic": nan2, "slope": float("nan"), "slope_ic": nan2}
    lp = _logit(p)
    intercept, slope = calibration_intercept_slope(y, p)
    z = norm.ppf(1 - (1 - conf) / 2)
    mu = 1 / (1 + np.exp(-(intercept + lp)))
    se_a = 1 / np.sqrt(np.sum(mu * (1 - mu)))
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    m = fit.predict_proba(lp.reshape(-1, 1))[:, 1]
    design = np.column_stack([np.ones_like(lp), lp])
    cov = np.linalg.inv(design.T @ (design * (m * (1 - m))[:, None]))
    se_b = float(np.sqrt(cov[1, 1]))
    return {
        "intercepto": intercept,
        "intercepto_ic": (intercept - z * se_a, intercept + z * se_a),
        "slope": slope,
        "slope_ic": (slope - z * se_b, slope + z * se_b),
    }


def oe_ci(y, p, conf: float = 0.95) -> tuple[float, float]:
    """IC exato de Poisson para O/E, tratando o número esperado como fixo."""
    from scipy.stats import chi2

    observed, expected = float(np.sum(y)), float(np.sum(p))
    tail = (1 - conf) / 2
    lo = 0.0 if observed == 0 else chi2.ppf(tail, 2 * observed) / 2
    hi = chi2.ppf(1 - tail, 2 * (observed + 1)) / 2
    return float(lo / expected), float(hi / expected)


def auroc_ci(y, p, conf: float = 0.95) -> tuple[float, float]:
    """IC aproximado de Hanley e McNeil (1982) para a AUROC."""
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return (float("nan"), float("nan"))
    a = roc_auc_score(y, p)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    z = norm.ppf(1 - (1 - conf) / 2)
    return float(max(0.0, a - z * se)), float(min(1.0, a + z * se))


def smoothed_calibration(y, p, n_knots: int = 5):
    """Curva de calibração flexível: logística de y sobre spline do logit de p. Devolve a função suavizada."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return lambda q: np.full(np.shape(np.atleast_1d(q)), np.nan)
    model = make_pipeline(SplineTransformer(n_knots=n_knots, degree=3), LogisticRegression(C=1e6, max_iter=2000))
    model.fit(_logit(p).reshape(-1, 1), y)
    return lambda q: model.predict_proba(_logit(q).reshape(-1, 1))[:, 1]


def calibration_curve_points(y, p, n_points: int = 50, n_knots: int = 5, n_boot: int = 0, seed: int = 42) -> dict:
    """Pontos da curva suavizada nos quantis de p, banda de 95% por bootstrap (se n_boot > 0) e histograma de p
    por desfecho (para a figura)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    grid = np.unique(np.quantile(p, np.linspace(0.005, 0.995, n_points)))
    edges = np.linspace(0.0, 1.0, 51)
    out = {
        "p": grid.tolist(),
        "observado_suavizado": smoothed_calibration(y, p, n_knots)(grid).tolist(),
        "bordas_histograma": edges.tolist(),
        "histograma_obitos": np.histogram(p[y == 1], bins=edges)[0].tolist(),
        "histograma_sobreviventes": np.histogram(p[y == 0], bins=edges)[0].tolist(),
    }
    if n_boot:
        rng = np.random.default_rng(seed)
        curves = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            curves.append(smoothed_calibration(y[i], p[i], n_knots)(grid))
        lo, hi = np.nanquantile(np.array(curves), [0.025, 0.975], axis=0)
        out["banda_inferior"], out["banda_superior"] = lo.tolist(), hi.tolist()
    return out


def integrated_calibration_index(y, p, n_knots: int = 5) -> float:
    """ICI: média de |p - curva suavizada| (Austin; Steyerberg, 2019), com spline no lugar do loess."""
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(p - smoothed_calibration(y, p, n_knots)(p))))


def bootstrap_ci(
    stat, n: int, n_boot: int = 1000, seed: int = 0, clusters=None, conf: float = 0.95
) -> tuple[float, float]:
    """IC percentil por bootstrap. stat recebe um vetor de índices. Com clusters, reamostra clusters inteiros."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        samples = (rng.integers(0, n, n) for _ in range(n_boot))
    else:
        clusters = np.asarray(clusters)
        labels = np.unique(clusters)
        members = [np.flatnonzero(clusters == c) for c in labels]
        samples = (
            np.concatenate([members[j] for j in rng.integers(0, len(labels), len(labels))]) for _ in range(n_boot)
        )
    values = np.array([stat(idx) for idx in samples], dtype=float)
    tail = (1 - conf) / 2
    return float(np.nanquantile(values, tail)), float(np.nanquantile(values, 1 - tail))


SURGE_RATIO = 3.0
SURGE_MIN_DEATHS = 10
SURGE_LAG_MONTHS = 3
SURGE_WINDOW_MONTHS = 12


def evaluate_period(y, p, released, alpha: float, r_hat: float, prevalence_cal: float, *, n1, groups=None) -> dict:
    """Discriminação, calibração e as duas garantias num período. alpha é sempre o nível nominal (meta de G1); n1
    são os óbitos da parte B por trás da regra (dicionário por grupo, com groups, na regra por grupo)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    g = evaluate_guarantees(y, released, alpha, r_hat, n1=n1, groups=groups)
    n_events = int(y.sum())
    both = 0 < n_events < len(y)
    nan2 = (float("nan"), float("nan"))
    cal = (
        calibration_intercept_slope_ci(y, p)
        if n_events >= 10 and both
        else {"intercepto": float("nan"), "intercepto_ic": nan2, "slope": float("nan"), "slope_ic": nan2}
    )
    return {
        "n": int(len(y)),
        "eventos": n_events,
        "prevalencia": float(y.mean()),
        "razao_prevalencia": float(y.mean() / prevalence_cal) if prevalence_cal > 0 else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auroc_ic": auroc_ci(y, p) if both else nan2,
        "oe": observed_expected(y, p),
        "oe_ic": oe_ci(y, p),
        "intercepto": cal["intercepto"],
        "intercepto_ic": cal["intercepto_ic"],
        "slope": cal["slope"],
        "slope_ic": cal["slope_ic"],
        "fracao_liberada": g["released_fraction"],
        "fdl": g["g1"]["value"],
        "fdl_k": g["g1"]["k"],
        "fdl_n": g["g1"]["n"],
        "p_g1": g["g1"]["p_violation"],
        "g1_nulo": g["g1"]["null"],
        "tol": g["g2"]["value"],
        "tol_k": g["g2"]["k"],
        "tol_n": g["g2"]["n"],
        "r_hat": float(r_hat),
        "p_g2": g["g2"]["p_violation"],
    }


def surge_codes(ref_codes, period_codes, ratio: float = SURGE_RATIO, min_deaths: int = SURGE_MIN_DEATHS) -> set:
    """CIDs de 3 caracteres em surto: pelo menos min_deaths óbitos no período e pelo menos ratio vezes o esperado
    pela participação entre os óbitos de referência (código ausente na referência tem esperado zero)."""
    per = pd.Series(np.asarray(period_codes, dtype=str)).value_counts()
    if per.empty:
        return set()
    ref = pd.Series(np.asarray(ref_codes, dtype=str)).value_counts(normalize=True)
    expected = ref.reindex(per.index).fillna(0.0) * per.sum()
    return set(per.index[(per >= min_deaths) & (per >= ratio * expected)])


def residual_g1(y, released, codes, dates, start: pd.Timestamp, index, alpha: float, *, n1) -> dict:
    """G1 sem os óbitos de diagnósticos em surto no período que começa em start. A referência são os óbitos dos
    12 meses que terminam 3 meses antes de start (mesma janela da estratégia c). O nulo é o da regra (n1 óbitos na
    parte B), só para a regra global."""
    if isinstance(n1, dict):
        raise ValueError("G1 sem surtos só existe para a regra global (n1 escalar)")
    y = np.asarray(y, dtype=int)
    codes = np.asarray(codes, dtype=str)
    ref = in_window(
        dates,
        start - pd.DateOffset(months=SURGE_LAG_MONTHS + SURGE_WINDOW_MONTHS),
        start - pd.DateOffset(months=SURGE_LAG_MONTHS),
    ) & (y == 1)
    deaths = y[index] == 1
    surge = surge_codes(codes[ref], codes[index][deaths])
    keep = deaths & ~np.isin(codes[index], list(surge))
    k, n = int((np.asarray(released) & keep).sum()), int(keep.sum())
    return {
        "fdl_sem_surto": k / n if n else float("nan"),
        "fdl_sem_surto_k": k,
        "fdl_sem_surto_n": n,
        "p_g1_sem_surto": g1_pvalue(k, n, fdl_null(n1, alpha)),
        "cids_em_surto": sorted(surge),
    }


def evaluate_strategy_rows(rows: list[dict], y, alpha: float, codes=None, dates=None) -> list[dict]:
    """Converte a saída de run_strategy em linhas de métricas; com codes e dates, acrescenta G1 sem os óbitos
    de diagnósticos em surto (H1b)."""
    y = np.asarray(y, dtype=int)
    out = []
    for r in rows:
        m = evaluate_period(
            y[r["index"]],
            r["p"],
            r["released"],
            alpha,
            r["r_hat"],
            r["prevalencia_cal"],
            n1=r["n1"],
            groups=r.get("grupos"),
        )
        if codes is not None and not isinstance(r["n1"], dict):
            m.update(residual_g1(y, r["released"], codes, dates, r["inicio"], r["index"], alpha, n1=r["n1"]))
        out.append(
            {
                "trimestre": r["trimestre"],
                "estrategia": r["estrategia"],
                "alpha_usado": r["alpha_usado"],
                "fallback": r["fallback"],
                **m,
            }
        )
    return out


def bootstrap_metrics(y, p, released, clusters, n_boot: int = 1000, seed: int = 42) -> dict:
    """ICs por bootstrap, reamostrando internações e reamostrando hospitais, das métricas principais."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rel = np.asarray(released, dtype=bool)
    stats = {
        "auroc": lambda i: roc_auc_score(y[i], p[i]) if 0 < y[i].sum() < len(i) else float("nan"),
        "oe": lambda i: y[i].sum() / p[i].sum(),
        "fdl": lambda i: (rel[i] & (y[i] == 1)).sum() / max(int((y[i] == 1).sum()), 1),
        "tol": lambda i: (rel[i] & (y[i] == 1)).sum() / max(int(rel[i].sum()), 1),
        "fracao_liberada": lambda i: rel[i].mean(),
    }
    return {
        k: {
            "linhas": bootstrap_ci(f, len(y), n_boot, seed),
            "hospitais": bootstrap_ci(f, len(y), n_boot, seed, clusters=np.asarray(clusters)),
        }
        for k, f in stats.items()
    }


LEVEL = 0.05


def _g1_aggregate(rows, k_key, n_key) -> dict:
    """FDL somada nos períodos, com IC exato e o p-valor beta-binomial agregado (cada linha traz o nulo da sua
    regra em g1_nulo; as linhas de H1b usam o nulo do período, porque a regra é a mesma). Linha sem os parâmetros
    da beta só entra se não liberou óbitos."""
    k, n = int(sum(r[k_key] for r in rows)), int(sum(r[n_key] for r in rows))
    nulls = [None if r["g1_nulo"] is None else tuple(r["g1_nulo"]) for r in rows]
    if any(nl is None and r[k_key] > 0 for nl, r in zip(nulls, rows)):
        raise ValueError(
            "agregado de G1 indefinido: linha com óbitos liberados sem os parâmetros da beta (regra por "
            "grupo com óbitos em vários grupos ou resultado antigo)"
        )
    return {
        "k": k,
        "n": n,
        "valor": k / n if n else float("nan"),
        "ic": exact_ci(k, n),
        "p_violacao": g1_aggregate_pvalue(k, [r[n_key] for r in rows], nulls),
    }


def _g1_robustness(settings, p_key, k_key, n_key) -> dict:
    detail, refuted = {}, False
    for name, rows in settings.items():
        rej = holm([r[p_key] for r in rows])["reject"]
        agg = _g1_aggregate(rows, k_key, n_key)
        bad = any(rej) or agg["p_violacao"] < LEVEL
        refuted |= bad
        detail[name] = {
            "periodos_violados": [r.get("trimestre", i) for i, (r, x) in enumerate(zip(rows, rej)) if x],
            "agregado": agg,
            "refuta": bad,
        }
    return {"refutada": refuted, "cenarios": detail}


def h1(settings: dict[str, list[dict]], alpha: float) -> dict:
    """G1 resiste: refutada se, em algum cenário, houver violação após Holm ou FDL agregada acima de alfa."""
    return _g1_robustness(settings, "p_g1", "fdl_k", "fdl_n")


def h1b(settings: dict[str, list[dict]], alpha: float) -> dict:
    """Mecanismo: sem os óbitos de diagnósticos em surto, G1 não é violada (mesmos critérios de H1)."""
    return _g1_robustness(settings, "p_g1_sem_surto", "fdl_sem_surto_k", "fdl_sem_surto_n")


def h2(settings: dict[str, list[dict]], high: float = 1.2, low: float = 1.0) -> dict:
    """G2 é frágil. Refutada se (i) menos da metade dos períodos com razão de prevalência >= high violar G2
    (só com pelo menos 2 desses períodos), ou (ii) houver violação após Holm, ou TOL agregada acima do limite, em
    períodos com razão <= low, ou (iii) a correlação de Spearman entre a razão e TOL - r_hat não for positiva."""
    detail, refuted = {}, False
    for name, rows in settings.items():
        rej = holm([r["p_g2"] for r in rows])["reject"]
        hi = [x for r, x in zip(rows, rej) if r["razao_prevalencia"] >= high]
        lo_rows = [r for r in rows if r["razao_prevalencia"] <= low]
        lo_rej = [x for r, x in zip(rows, rej) if r["razao_prevalencia"] <= low]
        crit_i = len(hi) >= 2 and sum(hi) < len(hi) / 2
        if lo_rows:
            k = sum(r["tol_k"] for r in lo_rows)
            n = sum(r["tol_n"] for r in lo_rows)
            bound = sum(r["tol_n"] * r["r_hat"] for r in lo_rows) / n if n else 1.0
            p_agg = binom_greater_pvalue(k, n, bound)
        else:
            p_agg = 1.0
        crit_ii = any(lo_rej) or p_agg < LEVEL
        finite = [r for r in rows if np.isfinite(r["tol"]) and np.isfinite(r["r_hat"])]  # sem liberadas, TOL indefinida
        ratio = [r["razao_prevalencia"] for r in finite]
        excess = [r["tol"] - r["r_hat"] for r in finite]
        defined = len(finite) >= 3 and np.ptp(ratio) > 0 and np.ptp(excess) > 0
        rho = (
            float(spearmanr(ratio, excess).statistic) if defined else float("nan")
        )  # indefinida conta como não positiva
        crit_iii = not (rho > 0)
        bad = crit_i or crit_ii or crit_iii
        refuted |= bad
        detail[name] = {
            "violacoes_razao_alta": f"{sum(hi)}/{len(hi)}",
            "violacao_razao_baixa": crit_ii,
            "p_agregado_razao_baixa": p_agg,
            "spearman": rho,
            "criterios": [crit_i, crit_ii, crit_iii],
            "refuta": bad,
        }
    return {"refutada": refuted, "cenarios": detail}


def h3(pairs: list[tuple[dict, dict]]) -> dict:
    """Atualizar ajuda: pares (estática, trimestral) por trimestre, de X3 e X4 juntos. Refutada se nem o
    |O/E - 1| (Wilcoxon pareado unilateral) nem as violações de G2 (McNemar exato unilateral, violação =
    p_g2 < 0,05 no trimestre) favorecerem a recalibração trimestral."""
    a = np.array([abs(sa["oe"] - 1) for sa, _ in pairs])
    c = np.array([abs(sc["oe"] - 1) for _, sc in pairs])
    p_w = float(wilcoxon(c, a, alternative="less").pvalue) if np.any(c != a) else 1.0
    va = np.array([sa["p_g2"] < LEVEL for sa, _ in pairs])
    vc = np.array([sc["p_g2"] < LEVEL for _, sc in pairs])
    b01, b10 = int((va & ~vc).sum()), int((~va & vc).sum())
    p_m = float(binomtest(b01, b01 + b10, 0.5, alternative="greater").pvalue) if b01 + b10 else 1.0
    return {
        "refutada": not (p_w < LEVEL or p_m < LEVEL),
        "p_wilcoxon": p_w,
        "p_mcnemar": p_m,
        "violacoes_estatica": int(va.sum()),
        "violacoes_trimestral": int(vc.sum()),
        "n_trimestres": len(pairs),
    }


def h4(
    released_all: float,
    released_adult: float,
    diffs: dict[str, tuple[float, float]],
    min_all: float = 0.50,
    min_adult: float = 0.40,
) -> dict:
    """Utilidade: frações liberadas mínimas e IC 95% da diferença de fração liberada contra cada linha de base
    (modelo menos linha de base) inteiramente acima de zero."""
    fails = {"fracao_todas": released_all < min_all, "fracao_adultos": released_adult < min_adult}
    for name, (lo, _hi) in diffs.items():
        fails[f"vs_{name}"] = not (lo > 0)
    return {"refutada": any(fails.values()), "falhas": fails}


def h5(groups: dict[str, list[dict]], alpha: float, min_deaths: int = 0) -> dict:
    """Mondrian cruzado mantém G1 em cada faixa: refutada se houver violação após Holm entre faixas e períodos;
    o agregado por faixa é descritivo. Faixas com menos de min_deaths óbitos ficam descritivas."""
    tested = {g: rows for g, rows in groups.items() if sum(r["fdl_n"] for r in rows) >= min_deaths}
    flat = [(g, r) for g, rows in tested.items() for r in rows]
    rej = holm([r["p_g1"] for _, r in flat])["reject"] if flat else []
    detail, refuted = {}, False
    for g, rows in groups.items():
        agg = _g1_aggregate(rows, "fdl_k", "fdl_n")
        if g not in tested:
            detail[g] = {"agregado_descritivo": agg, "descritiva": True}
            continue
        bad = any(x for (gg, _), x in zip(flat, rej) if gg == g)
        refuted |= bad
        detail[g] = {"agregado_descritivo": agg, "refuta": bad}
    return {"refutada": refuted, "faixas": detail}
