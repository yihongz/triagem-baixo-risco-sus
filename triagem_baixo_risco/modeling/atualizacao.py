"""Estratégias de atualização da calibração e da regra de liberação ao longo do tempo.

Cada estratégia recebe as probabilidades brutas do modelo base (p_raw) para todas as linhas e devolve, a cada
trimestre de teste, as probabilidades calibradas, a decisão de liberar e os alvos das garantias. As janelas de
recalibração nunca incluem linhas de treino; com menos de MIN_DEATHS óbitos na janela, a estratégia reaproveita a
última calibração válida.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import beta

from triagem_baixo_risco.dataset import in_window, quarter_labels
from triagem_baixo_risco.modeling.predict import (
    ReleaseRule,
    apply_rule,
    clopper_pearson_upper,
    fit_release_rule,
    fraction_deaths_released,
)
from triagem_baixo_risco.modeling.train import Fitted, fit_window

STRATEGIES = ("a", "b", "c", "d", "e", "f", "g")
LAG_MONTHS = 3
WINDOW_MONTHS = 12
MIN_DEATHS = 50
DAYS_PER_QUARTER = 91.3125


def quarter_starts(first, last) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq="QS"))


def _months_before(ts: pd.Timestamp, k: int) -> pd.Timestamp:
    return ts - pd.DateOffset(months=k)


def window_mask(
    strategy: str, q: pd.Timestamp, dates: pd.Series, train_mask, lag: int = LAG_MONTHS, window: int = WINDOW_MONTHS
) -> np.ndarray:
    """Janela de recalibração das estratégias b, c, d e f para o trimestre que começa em q."""
    if strategy in ("b", "d"):
        return in_window(dates, pd.Timestamp(q.year - 1, 1, 1), pd.Timestamp(q.year, 1, 1)) & ~train_mask
    if strategy == "c":
        return in_window(dates, _months_before(q, lag + window), _months_before(q, lag)) & ~train_mask
    if strategy == "f":
        return (dates < _months_before(q, lag)).to_numpy() & ~train_mask
    raise ValueError(f"estratégia sem janela própria: {strategy}")


def soft_weights(dates: pd.Series, q: pd.Timestamp, lag: int = LAG_MONTHS, half_life: float = 4.0) -> np.ndarray:
    """Pesos 0,5^(idade/meia-vida), com a idade em trimestres contada do fim da janela disponível."""
    age = (_months_before(q, lag) - dates).dt.days.to_numpy() / DAYS_PER_QUARTER
    return np.where(age >= 0, 0.5 ** (age / half_life), 0.0)


def estimate_prevalence_acc(p_cal, y_cal, p_target, tau: float) -> float:
    """Prevalência sem rótulos por classify-and-count ajustado (equivale ao BBSE no caso binário)."""
    y_cal = np.asarray(y_cal, dtype=int)
    hat_cal, hat_t = np.asarray(p_cal) >= tau, np.asarray(p_target) >= tau
    tpr, fpr = hat_cal[y_cal == 1].mean(), hat_cal[y_cal == 0].mean()
    if tpr - fpr < 1e-6:
        return float(y_cal.mean())
    return float(np.clip((hat_t.mean() - fpr) / (tpr - fpr), 1e-6, 1 - 1e-6))


def prior_shift(p, pi_old: float, pi_new: float) -> np.ndarray:
    """Corrige probabilidades calibradas para uma nova prevalência, com o mesmo P(X | Y)."""
    p = np.asarray(p, dtype=float)
    r1, r0 = pi_new / pi_old, (1 - pi_new) / (1 - pi_old)
    return r1 * p / (r1 * p + r0 * (1 - p))


def adjusted_rate_bound(fitted: Fitted, rule: ReleaseRule, pi_new: float, conf: float = 0.95) -> float:
    """Limite de G2 recalculado para a prevalência pi_new, a partir de limites unilaterais conservadores estimados
    na parte B: superior para P(liberar | óbito) e inferior para P(liberar | sobrevivência)."""
    rel = apply_rule(rule, fitted.p_B)
    y = fitted.y_B
    k1, n1 = int((rel & (y == 1)).sum()), int((y == 1).sum())
    k0, n0 = int((rel & (y == 0)).sum()), int((y == 0).sum())
    a_up = clopper_pearson_upper(k1, n1, conf)
    b_lo = 0.0 if k0 == 0 else float(beta.ppf(1 - conf, k0, n0 - k0 + 1))
    if b_lo <= 0:
        return 1.0
    return float(pi_new * a_up / (pi_new * a_up + (1 - pi_new) * b_lo))


def run_strategy(
    strategy: str,
    *,
    p_raw,
    y,
    dates: pd.Series,
    quarters,
    base_mask,
    train_mask,
    kind: str = "platt",
    alpha: float = 0.05,
    seed: int = 42,
    lag: int = LAG_MONTHS,
    window: int = WINDOW_MONTHS,
    gamma: float = 0.02,
    half_life: float = 4.0,
    refit: Callable[[pd.Timestamp], np.ndarray] | None = None,
    groups=None,
    conf: float = 0.95,
) -> list[dict]:
    """Aplica a estratégia a cada trimestre de teste. refit(corte) só é usado pela estratégia d e devolve as
    probabilidades brutas de um modelo treinado com as internações anteriores ao corte."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy}")
    if strategy in ("e", "g") and groups is not None:
        raise ValueError("as estratégias e e g só existem para a regra global")
    y = np.asarray(y, dtype=int)
    train_mask = np.asarray(train_mask, dtype=bool)
    fit = dict(kind=kind, alpha=alpha, seed=seed, groups=groups, conf=conf)
    base = fit_window(p_raw, y, base_mask, **fit)
    last, cache, fdl_hist, alpha_t, out = base, {}, {}, alpha, []

    def fit_or_last(p_src, mask, weights=None):
        if int(y[mask].sum()) < MIN_DEATHS:
            return last, True
        return fit_window(p_src, y, mask, weights=weights, **fit), False

    for i, q in enumerate(quarters):
        test = in_window(dates, q, q + pd.DateOffset(months=3))
        p_src, fallback = p_raw, False
        if strategy in ("a", "e", "g"):
            fitted = base
        elif strategy == "d":
            if q.year not in cache:
                cutoff = pd.Timestamp(q.year - 1, 1, 1)
                p_new = refit(cutoff)
                # janela do ano anterior, sem linhas de treino
                cal_mask = window_mask("d", q, dates, train_mask, lag, window)
                cache[q.year] = (*fit_or_last(p_new, cal_mask), p_new)
            fitted, fallback, p_src = cache[q.year]
        else:
            key = (q.year,) if strategy == "b" else (q,)
            if key not in cache:
                mask = window_mask(strategy, q, dates, train_mask, lag, window)
                weights = soft_weights(dates, q, lag, half_life) if strategy == "f" else None
                cache[key] = fit_or_last(p_raw, mask, weights)
            fitted, fallback = cache[key]
        last = fitted

        p_t = fitted.calibrate(p_src[test])
        rule, alpha_used, prevalence_cal = fitted.rule, alpha, fitted.prevalence
        if strategy == "e":
            if i >= 2 and (i - 2) in fdl_hist and not np.isnan(fdl_hist[i - 2]):
                # com atraso de 3 meses, o trimestre rotulado mais recente é i-2
                alpha_t = alpha_t + gamma * (alpha - fdl_hist[i - 2])
            alpha_used = float(np.clip(alpha_t, 0.0, 0.5))
            rule = fit_release_rule(base.p_B, base.y_B, alpha_used, conf)
        released = apply_rule(rule, p_t, None if groups is None else np.asarray(groups)[test])
        r_hat, p_eval = rule.r_hat, p_t
        if strategy == "g":
            pi_hat = estimate_prevalence_acc(base.p_B, base.y_B, p_t, tau=base.prevalence)
            p_eval = prior_shift(p_t, base.prevalence, pi_hat)
            r_hat = adjusted_rate_bound(base, rule, pi_hat, conf)
        fdl_hist[i] = fraction_deaths_released(y[test], released)["value"]
        row = {
            "trimestre": quarter_labels(pd.Series([q])).iloc[0],
            "inicio": q,
            "estrategia": strategy,
            "index": np.flatnonzero(test),
            "p": p_eval,
            "released": released,
            "alpha_usado": alpha_used,
            "r_hat": r_hat,
            "prevalencia_cal": prevalence_cal,
            "fallback": fallback,
            "n1": rule.n1,
        }
        if groups is not None:
            row["grupos"] = np.asarray(groups)[test]
        out.append(row)
    return out
