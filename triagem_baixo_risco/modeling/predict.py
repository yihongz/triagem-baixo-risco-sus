"""Conjuntos de predição conformais, a regra de liberação de baixo risco e a banda de custo com abstenção.

G1 (por classe): no máximo alfa dos óbitos são liberados. G2 (de taxa): a taxa de óbito entre as liberadas fica
abaixo do limite superior r_hat estimado na parte B da calibração. A regra libera quando o rótulo óbito fica fora do
conjunto conformal por classe, comparando escores, nunca um limiar recalculado em p (que erra nos empates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import beta, betabinom, binom, binomtest


def conformal_quantile(scores, alpha: float) -> float:
    """Quantil com correção de amostra finita: k-ésimo menor escore, k = ceil((n+1)(1-alpha))."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("conjunto de calibração vazio")
    k = math.ceil((n + 1) * (1.0 - alpha) - 1e-12)
    if k > n:
        return float("inf")
    return float(s[k - 1])


def prediction_sets_class_conditional(probs, quantiles_by_class: dict):
    """Limiar por rótulo candidato (Mondrian por classe): protege a cobertura da classe rara."""
    probs = np.asarray(probs, dtype=float)
    q = np.array([quantiles_by_class[k] for k in range(probs.shape[1])], dtype=float)
    return (1.0 - probs) <= q[None, :]


def death_scores(p) -> np.ndarray:
    """Escore de não conformidade do rótulo óbito: 1 - p."""
    return 1.0 - np.asarray(p, dtype=float)


def weighted_conformal_quantile(scores, weights, alpha: float, test_weight: float | None = None) -> float:
    """Quantil 1 - alfa da distribuição ponderada dos escores, com massa em +inf para o ponto de teste
    (Tibshirani et al., 2019; Barber et al., 2023). Com pesos iguais, coincide com conformal_quantile."""
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(s) == 0 or w.sum() <= 0:
        return float("inf")  # sem massa na calibração: não libera ninguém
    if test_weight is None:
        test_weight = float(w.max())
    order = np.argsort(s, kind="stable")
    s, w = s[order], w[order]
    cum = np.cumsum(w) / (w.sum() + test_weight)
    idx = int(np.searchsorted(cum, 1.0 - alpha - 1e-12, side="left"))
    return float(s[idx]) if idx < len(s) else float("inf")


def clopper_pearson_upper(k: float, n: float, conf: float = 0.95) -> float:
    """Limite superior unilateral exato de uma proporção (aceita k e n não inteiros, para amostras ponderadas)."""
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def exact_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Intervalo bilateral exato de uma proporção (Clopper-Pearson)."""
    if n == 0:
        return (float("nan"), float("nan"))
    tail = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lo, hi


def binom_greater_pvalue(k: int, n: int, p0: float) -> float:
    """p-valor do teste binomial exato unilateral: H0 proporção <= p0 contra H1 proporção > p0."""
    if n == 0 or p0 >= 1.0:
        return 1.0
    if p0 <= 0.0:
        return 0.0 if k > 0 else 1.0
    return float(binomtest(int(k), int(n), float(p0), alternative="greater").pvalue)


def holm(pvalues, level: float = 0.05) -> dict:
    """Correção de Holm: p-valores ajustados e rejeições ao nível dado."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return {"p_adjusted": adj.tolist(), "reject": (adj <= level).tolist()}


@dataclass(frozen=True)
class ReleaseRule:
    """Regra de liberação ajustada na parte B: quantil(s) de corte e o limite superior da taxa de óbito."""

    alpha: float
    q1: float | dict  # quantil global ou por grupo (Mondrian cruzado)
    r_hat: float  # limite superior de 95% da taxa de óbito entre as liberadas, estimado na parte B
    n_released_cal: float
    deaths_released_cal: float
    by_group: bool = False
    n1: float | dict | None = None  # óbitos da parte B por trás de q1 (tamanho efetivo de Kish, com pesos)


def _quantile(scores, alpha, weights=None) -> float:
    if len(scores) == 0:
        return float("inf")  # sem óbitos na calibração: não libera ninguém
    if weights is None:
        return conformal_quantile(scores, alpha)
    return weighted_conformal_quantile(scores, weights, alpha)


def _n_deaths(mask, w) -> float:
    """Óbitos por trás de q1: a contagem ou, com pesos, o tamanho efetivo de Kish, (soma w)² / soma w²."""
    if w is None:
        return int(mask.sum())
    wm = w[mask]
    return float(wm.sum() ** 2 / (wm**2).sum()) if wm.sum() > 0 else 0.0


def fit_release_rule(p_cal, y_cal, alpha: float, conf: float = 0.95, groups_cal=None, weights_cal=None) -> ReleaseRule:
    """Ajusta a regra na parte B da calibração: quantil por classe (ou por classe e grupo) e o limite r_hat."""
    s = death_scores(p_cal)
    y = np.asarray(y_cal, dtype=int)
    w = None if weights_cal is None else np.asarray(weights_cal, dtype=float)
    deaths = y == 1
    if groups_cal is None:
        q1 = _quantile(s[deaths], alpha, None if w is None else w[deaths])
        n1 = _n_deaths(deaths, w)
        released = s > q1
    else:
        g = np.asarray(groups_cal)
        q1 = {
            lv: _quantile(s[deaths & (g == lv)], alpha, None if w is None else w[deaths & (g == lv)])
            for lv in np.unique(g)
        }
        n1 = {lv: _n_deaths(deaths & (g == lv), w) for lv in np.unique(g)}
        released = s > np.array([q1[v] for v in g], dtype=float)
    if w is None:
        n, k = float(released.sum()), float((released & deaths).sum())
        r_hat = clopper_pearson_upper(int(k), int(n), conf)
    else:
        wr = w[released]
        n = float(wr.sum() ** 2 / (wr**2).sum()) if wr.sum() > 0 else 0.0  # tamanho efetivo de amostra
        k = float((wr * y[released]).sum() / wr.sum() * n) if wr.sum() > 0 else 0.0
        r_hat = clopper_pearson_upper(k, n, conf)
    return ReleaseRule(
        alpha=alpha,
        q1=q1,
        r_hat=r_hat,
        n_released_cal=n,
        deaths_released_cal=k,
        by_group=groups_cal is not None,
        n1=n1,
    )


def apply_rule(rule: ReleaseRule, p, groups=None) -> np.ndarray:
    """True = liberada. Grupo sem quantil na calibração nunca é liberado."""
    s = death_scores(p)
    if not rule.by_group:
        return s > rule.q1
    if groups is None:
        raise ValueError("a regra por grupo exige groups")
    q = np.array([rule.q1.get(v, np.inf) for v in np.asarray(groups)], dtype=float)
    return s > q


def fraction_deaths_released(y, released) -> dict:
    """G1: fração dos óbitos que a regra liberou."""
    y = np.asarray(y, dtype=int)
    rel = np.asarray(released, dtype=bool)
    n, k = int((y == 1).sum()), int((rel & (y == 1)).sum())
    return {"value": k / n if n else float("nan"), "k": k, "n": n}


def death_rate_released(y, released) -> dict:
    """G2: taxa de óbito entre as liberadas."""
    y = np.asarray(y, dtype=int)
    rel = np.asarray(released, dtype=bool)
    n, k = int(rel.sum()), int((rel & (y == 1)).sum())
    return {"value": k / n if n else float("nan"), "k": k, "n": n}


def fdl_null(n1, alpha: float):
    """Parâmetros (a, b) da beta que descreve, sob permutabilidade, a fração de óbitos liberados dada a
    calibração: exata quando n1 é inteiro (o k-ésimo menor escore, k = ceil((n1 + 1)(1 - alfa))), contínua
    quando n1 é o tamanho efetivo com pesos. Devolve None quando a regra não libera nenhum óbito."""
    n1 = float(n1)
    if n1 <= 0:
        return None
    if n1.is_integer():
        k = math.ceil((n1 + 1) * (1.0 - alpha) - 1e-12)
        a = n1 + 1 - k
        return None if a <= 0 else (float(a), float(k))
    return (alpha * (n1 + 1), (1.0 - alpha) * (n1 + 1))


def g1_pvalue(k: int, m: int, null) -> float:
    """p-valor unilateral de G1: P(X >= k) com X ~ BetaBinom(m, a, b), a distribuição do número de óbitos liberados
    entre m óbitos novos (Angelopoulos; Bates, 2022). Ela continua exata quando os escores dos óbitos são só
    permutáveis com os da calibração, porque depende apenas da ordem entre eles."""
    if m == 0 or k <= 0:
        return 1.0
    if null is None:
        return 0.0  # a regra não libera óbitos; liberar algum contradiz o nulo
    return float(betabinom.sf(int(k) - 1, int(m), *null))


def _sum_pmf(pmfs) -> np.ndarray:
    out = np.array([1.0])
    for p in pmfs:
        out = np.clip(fftconvolve(out, p), 0.0, None)
    return out / out.sum()


def _tail(pmf: np.ndarray, k: int) -> float:
    return float(pmf[k:].sum()) if k < len(pmf) else 0.0


def _point_mass_zero(m: int) -> np.ndarray:
    return np.r_[1.0, np.zeros(m)]


def g1_group_pvalue(ks, ms, nulls) -> float:
    """G1 de uma regra por grupo (Mondrian cruzado): P(soma dos X_g >= soma dos k_g), com X_g ~ BetaBinom(m_g,
    a_g, b_g) independentes, porque as calibrações dos grupos são disjuntas."""
    k = int(sum(ks))
    if k <= 0:
        return 1.0
    pmfs = [
        _point_mass_zero(int(m)) if nl is None else betabinom.pmf(np.arange(int(m) + 1), int(m), *nl)
        for m, nl in zip(ms, nulls)
        if m > 0
    ]
    return _tail(_sum_pmf(pmfs), k) if pmfs else 1.0


def g1_aggregate_pvalue(k: int, ms, nulls, n_nodes: int = 128) -> float:
    """G1 agregado sobre períodos: P(soma dos X_i >= k). Com uma calibração só, os parâmetros são iguais em
    todo período e a soma é BetaBinom exata. Com calibrações diferentes, supõe dependência máxima entre elas
    (FDLs comonotônicas, F_i^-1(u) com u uniforme) e integra em u por Gauss-Legendre."""
    items = [(int(m), nl) for m, nl in zip(ms, nulls) if m > 0]
    if k <= 0 or not items:
        return 1.0
    if len({nl for _, nl in items}) == 1:
        return g1_pvalue(k, sum(m for m, _ in items), items[0][1])
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    total = 0.0
    for u, wu in zip((x + 1) / 2, w / 2):
        pmfs = [
            _point_mass_zero(m) if nl is None else binom.pmf(np.arange(m + 1), m, beta.ppf(u, *nl)) for m, nl in items
        ]
        total += wu * _tail(_sum_pmf(pmfs), int(k))
    return float(min(1.0, total))


def evaluate_guarantees(y, released, alpha: float, r_hat: float, conf: float = 0.95, *, n1, groups=None) -> dict:
    """Frações observadas, ICs exatos e p-valores de violação de G1 (beta-binomial sob permutabilidade, com os
    n1 óbitos da parte B) e G2 (binomial contra r_hat). Numa regra por grupo, n1 é um dicionário e groups traz o
    grupo de cada linha. g1["null"] é o (a, b) do nulo de G1, ou None quando ele não se resume a uma única beta."""
    y = np.asarray(y, dtype=int)
    rel = np.asarray(released, dtype=bool)
    g1, g2 = fraction_deaths_released(y, rel), death_rate_released(y, rel)
    if isinstance(n1, dict):
        if groups is None:
            raise ValueError("a regra por grupo exige groups")
        g, dead = np.asarray(groups), y == 1
        levels = np.unique(g[dead])
        nulls = [fdl_null(n1.get(lv, 0), alpha) for lv in levels]
        p1 = g1_group_pvalue(
            [int((rel & dead & (g == lv)).sum()) for lv in levels],
            [int((dead & (g == lv)).sum()) for lv in levels],
            nulls,
        )
        null = nulls[0] if len(levels) == 1 else None  # óbitos de um grupo só: o nulo é a beta desse grupo
    else:
        null = fdl_null(n1, alpha)
        p1 = g1_pvalue(g1["k"], g1["n"], null)
    g1.update(
        ci=exact_ci(g1["k"], g1["n"], conf), target=alpha, p_violation=p1, null=None if null is None else list(null)
    )
    g2.update(
        ci=exact_ci(g2["k"], g2["n"], conf), target=r_hat, p_violation=binom_greater_pvalue(g2["k"], g2["n"], r_hat)
    )
    return {"released_fraction": float(np.mean(rel)), "g1": g1, "g2": g2}


DEFER = -1


def expected_cost(y, decision, c_fn: float, c_fp: float, c_defer: float) -> float:
    """Custo esperado por caso: média ponderada de falsos negativos, falsos positivos e encaminhamentos."""
    y = np.asarray(y, dtype=int)
    d = np.asarray(decision, dtype=int)
    fn = ((d == 0) & (y == 1)).sum()
    fp = ((d == 1) & (y == 0)).sum()
    deferred = (d == DEFER).sum()
    return float((c_fn * fn + c_fp * fp + c_defer * deferred) / len(y))


def decide_with_band(p1, t_lo: float, t_hi: float):
    """Regra de Chow com custos assimétricos: positivo se p >= t_hi, negativo se p <= t_lo, senão encaminha."""
    p1 = np.asarray(p1, dtype=float)
    return np.where(p1 >= t_hi, 1, np.where(p1 <= t_lo, 0, DEFER))


def choose_band(y, p1, grid, c_fn: float, c_fp: float, c_defer: float):
    """Par (t_lo, t_hi) de menor custo esperado; empate favorece a banda mais estreita e depois o menor t_lo."""
    y = np.asarray(y, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    grid = sorted(float(g) for g in grid)
    best, best_cost = None, np.inf
    for i, t_lo in enumerate(grid):
        for t_hi in grid[i:]:
            cost = expected_cost(y, decide_with_band(p1, t_lo, t_hi), c_fn, c_fp, c_defer)
            key = (cost, t_hi - t_lo, t_lo)
            # o "or" só lê best_key depois da primeira atribuição; best_cost não é usado
            if best is None or key < best_key:  # noqa: F821
                best, best_key, best_cost = (t_lo, t_hi), key, cost  # noqa: F841
    return best
