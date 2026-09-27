"""Gerador de linhas sintéticas no formato bruto do SIH-SUS (RD), só para testes e demonstração."""

from __future__ import annotations

import numpy as np
import pandas as pd

_CIDS = ["I219", "J189", "K359", "O800", "S720", "C509", "N390", "E119", "F200", "A090"]
_PROCS = ["0303060212", "0303140151", "0407040064", "0310010039", "0408050128", "0303010037"]
_MUNS = ["314330", "310120", "311340", "313940", "312150"]


def synthetic_raw(
    n: int = 1000,
    seed: int = 0,
    years=(2019, 2020, 2021, 2022, 2023, 2024, 2025),
    logit_shift: dict | None = None,
    cmpt_delay_max: int = 0,
) -> pd.DataFrame:
    """logit_shift: {ano: deslocamento no logit do óbito}, para simular deriva de prevalência entre anos.
    cmpt_delay_max: competência sorteada entre o mês da internação e esse número de meses depois (sorteio feito por
    último, para não mudar as outras colunas); com 0, a competência é o mês da internação."""
    rng = np.random.default_rng(seed)
    idade = rng.integers(0, 95, n)
    urgencia = rng.random(n) < 0.7
    cid = rng.choice(_CIDS, n)
    grave = np.isin(cid, ["I219", "J189", "C509"])
    ano = rng.choice(years, n)
    shift = np.array([(logit_shift or {}).get(int(a), 0.0) for a in ano])
    logit = -4.0 + 0.045 * idade + 0.8 * urgencia + 1.0 * grave + shift
    p_morte = 1 / (1 + np.exp(-logit))
    morte = rng.random(n) < p_morte
    dias = np.clip(rng.poisson(3 + 0.05 * idade + 4 * grave), 0, 120)
    mes = rng.integers(1, 13, n)
    dia = rng.integers(1, 29, n)
    inter = pd.to_datetime(pd.DataFrame({"year": ano, "month": mes, "day": dia}))
    saida = inter + pd.to_timedelta(dias, unit="D")
    nasc = inter - pd.to_timedelta(idade * 365 + rng.integers(0, 365, n), unit="D")
    mun_mov = rng.choice(_MUNS, n)
    mun_res = np.where(rng.random(n) < 0.7, mun_mov, rng.choice(_MUNS, n))
    outros = rng.choice(["12", "12", "12", "12", "31", "27"], n)
    cobranca = np.where(morte, "41", np.where(cid == "O800", "61", outros))
    raw = pd.DataFrame(
        {
            "ANO_CMPT": ano.astype(str),
            "MES_CMPT": [f"{m:02d}" for m in mes],
            "IDADE": idade.astype(str),
            "COD_IDADE": np.where(idade == 0, "3", "4"),
            "SEXO": rng.choice(["1", "3"], n),
            "RACA_COR": rng.choice(["01", "03", "99"], n),
            "MUNIC_RES": mun_res,
            "MUNIC_MOV": mun_mov,
            "CNES": rng.choice(["2200001", "2200002", "2200003"], n),
            "ESPEC": rng.choice(["01", "03", "05"], n),
            "CAR_INT": np.where(urgencia, "02", "01"),
            "DIAG_PRINC": cid,
            "PROC_SOLIC": rng.choice(_PROCS, n),
            "COMPLEX": rng.choice(["02", "03"], n),
            "DT_INTER": inter.dt.strftime("%Y%m%d").to_numpy(),
            "DT_SAIDA": saida.dt.strftime("%Y%m%d").to_numpy(),
            "NASC": nasc.dt.strftime("%Y%m%d").to_numpy(),
            "CEP": rng.choice([f"39{k:06d}" for k in range(400)], n),
            "DIAS_PERM": dias.astype(str),
            "UTI_MES_TO": rng.integers(0, 5, n).astype(str),
            "VAL_TOT": rng.gamma(2, 800, n).round(2).astype(str),
            "MORTE": np.where(morte, "1", "0"),
            "COBRANCA": cobranca,
            "PROC_REA": rng.choice(_PROCS, n),
            "N_AIH": [f"31{seed:04d}{i:07d}" for i in range(n)],
            "IDENT": "1",
        }
    )
    if cmpt_delay_max:
        month = ano * 12 + mes - 1 + rng.integers(0, cmpt_delay_max + 1, n)
        raw["ANO_CMPT"] = (month // 12).astype(str)
        raw["MES_CMPT"] = [f"{m % 12 + 1:02d}" for m in month]
    return raw
