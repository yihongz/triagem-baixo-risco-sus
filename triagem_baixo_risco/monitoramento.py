"""Monitoramento da deriva (X8): controle binomial do O/E e martingale de permutabilidade (Vovk et al., 2005)."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def oe_control(observed: float, expected: float, n: int, level: float = 0.95) -> dict:
    """Limites binomiais para O/E, com eventos ~ Binomial(n, esperado/n); alarme se O/E cair fora deles."""
    q = expected / n
    lo = binom.ppf((1 - level) / 2, n, q) / expected
    hi = binom.ppf(1 - (1 - level) / 2, n, q) / expected
    oe = observed / expected
    return {"oe": float(oe), "lim_inf": float(lo), "lim_sup": float(hi), "alarme": bool(not lo <= oe <= hi)}


def conformal_pvalues(cal_scores, test_scores, seed: int = 0) -> np.ndarray:
    """p-valores conformais suavizados: (#cal > s + U (#cal = s + 1)) / (n + 1). Uniformes sob permutabilidade."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    s = np.asarray(test_scores, dtype=float)
    n = len(cal)
    right = np.searchsorted(cal, s, side="right")
    left = np.searchsorted(cal, s, side="left")
    u = np.random.default_rng(seed).random(len(s))
    return (n - right + u * (right - left + 1)) / (n + 1)


def mixture_power_martingale_log10(pvalues, eps=None, chunk: int = 10000) -> np.ndarray:
    """log10 do martingale de potência misturado em epsilon (média sobre a grade). A desigualdade de Ville daria
    P(sup M >= 100) <= 0,01 para p-valores conformais válidos e independentes; aqui eles compartilham o conjunto de
    calibração, então esse limite não é garantido e os alarmes do X8 são descritivos."""
    eps = np.linspace(0.01, 0.99, 99) if eps is None else np.asarray(eps, dtype=float)
    cum = np.cumsum(np.log(np.clip(np.asarray(pvalues, dtype=float), 1e-12, 1.0)))
    out = np.empty(len(cum))
    log_eps = np.log(eps)
    for a in range(0, len(cum), chunk):
        t = np.arange(a + 1, min(a + chunk, len(cum)) + 1)[:, None]
        logm = t * log_eps[None, :] + (eps[None, :] - 1.0) * cum[a : a + chunk, None]
        top = logm.max(axis=1, keepdims=True)
        out[a : a + chunk] = (top[:, 0] + np.log(np.exp(logm - top).mean(axis=1))) / np.log(10)
    return out


def first_alarm(log10_m, threshold: float = 2.0) -> int | None:
    idx = np.flatnonzero(np.asarray(log10_m) >= threshold)
    return int(idx[0]) if len(idx) else None
