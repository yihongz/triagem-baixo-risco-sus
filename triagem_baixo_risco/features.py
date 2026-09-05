"""Variáveis conhecidas na admissão, estratos para relatar resultados por subgrupo e ligação de AIH em episódios."""

from __future__ import annotations

import pandas as pd

# Colunas que só existem depois da admissão, ou que são o próprio desfecho: nunca entram em X.
LEAKY_COLUMNS = frozenset(
    {
        "DIAS_PERM",
        "QT_DIARIAS",
        "UTI_MES_TO",
        "MARCA_UTI",
        "UTI_INT_TO",
        "DIAR_ACOM",
        "VAL_SH",
        "VAL_SP",
        "VAL_TOT",
        "VAL_UTI",
        "VAL_UCI",
        "US_TOT",
        "TOT_PT_SP",
        "VAL_SH_FED",
        "VAL_SP_FED",
        "VAL_SH_GES",
        "VAL_SP_GES",
        "MARCA_UCI",
        "DT_SAIDA",
        "COBRANCA",
        "MORTE",
        "CID_MORTE",
        "PROC_REA",
        "INFEHOSP",
        "DIAG_SECUN",
        "CID_ASSO",
        "IND_VDRL",
        # em 99,6% das AIH, COMPLEX é determinada pelo procedimento realizado: vaza informação pós-admissão
        "COMPLEX",
        *[f"DIAGSEC{i}" for i in range(1, 10)],
        *[f"TPDISEC{i}" for i in range(1, 10)],
    }
)

# Colunas brutas de onde saem as variáveis do modelo. RACA_COR fica de fora por outro motivo, a mudança de codificação:
# "sem informação" ficou entre 28% e 33% ao ano de 2019 a 2021, caiu em 2022 e é zero desde o 2º trimestre de 2023.
SOURCE_COLUMNS = (
    "IDADE",
    "COD_IDADE",
    "SEXO",
    "CAR_INT",
    "ESPEC",
    "DIAG_PRINC",
    "PROC_SOLIC",
    "MUNIC_RES",
    "MUNIC_MOV",
    "CNES",
    "DT_INTER",
    "ANO_CMPT",
)

ADMISSION_FEATURES = [
    "idade_anos",
    "sexo",
    "carater",
    "urgencia",
    "especialidade",
    "cid_capitulo",
    "cid3",
    "proc_grupo",
    "proc_subgrupo",
    "munic_res",
    "munic_mov",
    "mesmo_municipio",
    "cnes",
    "mes_internacao",
    "dia_semana_internacao",
    "ano_cmpt",
]

AGE_BANDS = ["0-17", "18-39", "40-59", "60-79", "80+"]
POLO_MUNICIPIO = "314330"  # Montes Claros

_ICD10_CHAPTERS = [
    ("A00", "B99", "I"),
    ("C00", "D48", "II"),
    ("D50", "D89", "III"),
    ("E00", "E90", "IV"),
    ("F00", "F99", "V"),
    ("G00", "G99", "VI"),
    ("H00", "H59", "VII"),
    ("H60", "H95", "VIII"),
    ("I00", "I99", "IX"),
    ("J00", "J99", "X"),
    ("K00", "K93", "XI"),
    ("L00", "L99", "XII"),
    ("M00", "M99", "XIII"),
    ("N00", "N99", "XIV"),
    ("O00", "O99", "XV"),
    ("P00", "P96", "XVI"),
    ("Q00", "Q99", "XVII"),
    ("R00", "R99", "XVIII"),
    ("S00", "T98", "XIX"),
    ("V01", "Y98", "XX"),
    ("Z00", "Z99", "XXI"),
    ("U00", "U99", "XXII"),
]


def age_in_years(idade, cod_idade) -> int | None:
    """COD_IDADE do SIH: 2 = dias, 3 = meses, 4 = anos, 5 = acima de 100 anos, 0 = ignorado."""
    try:
        idade, cod = int(idade), int(cod_idade)
    except (TypeError, ValueError):
        return None
    if cod == 2:
        return idade // 365
    if cod == 3:
        return idade // 12
    if cod == 4:
        return idade
    if cod == 5:
        return 100 + idade
    return None


def icd10_chapter(code: str | None) -> str | None:
    """Capítulo da CID-10 (algarismo romano) a partir dos três primeiros caracteres do código."""
    if not code or len(code) < 3:
        return None
    c = code[:3].upper()
    for lo, hi, chapter in _ICD10_CHAPTERS:
        if lo <= c <= hi:
            return chapter
    return None


def _col(df: pd.DataFrame, name: str, default: str = "NA") -> pd.Series:
    """Coluna como texto (com strip), ou uma série constante com default quando a coluna não existe."""
    if name in df.columns:
        return df[name].astype(str).str.strip()
    return pd.Series(default, index=df.index)


def build_features(raw: pd.DataFrame, prolonged_days: int = 8):
    """Retorna (X, y): X só com variáveis conhecidas na admissão; y com 'morte' e 'perm_prolongada'."""
    X = pd.DataFrame(index=raw.index)
    X["idade_anos"] = [age_in_years(i, c) for i, c in zip(raw["IDADE"], raw["COD_IDADE"])]
    X["sexo"] = _col(raw, "SEXO")
    carater = _col(raw, "CAR_INT").str.zfill(2)
    X["carater"] = carater
    X["urgencia"] = (carater != "01").astype(int)
    X["especialidade"] = _col(raw, "ESPEC").str.zfill(2)
    cid = _col(raw, "DIAG_PRINC").str.upper()
    X["cid_capitulo"] = cid.map(icd10_chapter)
    X["cid3"] = cid.str[:3]
    proc = _col(raw, "PROC_SOLIC").str.zfill(10)
    X["proc_grupo"] = proc.str[:2]
    X["proc_subgrupo"] = proc.str[:4]
    X["munic_res"] = _col(raw, "MUNIC_RES")
    X["munic_mov"] = _col(raw, "MUNIC_MOV")
    X["mesmo_municipio"] = (X["munic_res"] == X["munic_mov"]).astype(int)
    X["cnes"] = _col(raw, "CNES")
    dt = pd.to_datetime(_col(raw, "DT_INTER"), format="%Y%m%d", errors="coerce")
    X["mes_internacao"] = dt.dt.month
    X["dia_semana_internacao"] = dt.dt.dayofweek
    X["ano_cmpt"] = pd.to_numeric(_col(raw, "ANO_CMPT"), errors="coerce").astype("Int64")

    y = pd.DataFrame(index=raw.index)
    y["morte"] = (_col(raw, "MORTE") == "1").astype(int)
    y["perm_prolongada"] = (pd.to_numeric(_col(raw, "DIAS_PERM"), errors="coerce") >= prolonged_days).astype(int)

    assert set(X.columns).isdisjoint(LEAKY_COLUMNS), "vazamento: coluna pós-admissão em X"
    return X[ADMISSION_FEATURES], y


def build_strata(raw: pd.DataFrame) -> pd.DataFrame:
    """Marcadores para estratificar resultados e definir subpopulações. Não entram no modelo."""
    idade = pd.Series(
        [age_in_years(i, c) for i, c in zip(raw["IDADE"], raw["COD_IDADE"])], index=raw.index, dtype="float"
    )
    cob = _col(raw, "COBRANCA")
    capitulo = _col(raw, "DIAG_PRINC").str.upper().map(icd10_chapter)
    s = pd.DataFrame(index=raw.index)
    s["idade"] = idade
    s["censurada"] = cob.isin([str(c) for c in range(21, 30)]) | cob.isin(["31", "32", "51"])
    s["obstetrica"] = (capitulo == "XV") | cob.isin([str(c) for c in range(61, 68)])
    s["adulto"] = idade >= 18
    s["urgencia"] = _col(raw, "CAR_INT").str.zfill(2) != "01"
    faixa = pd.cut(idade, bins=[-1, 17, 39, 59, 79, 1000], labels=AGE_BANDS)
    s["faixa_etaria"] = faixa.astype(str).replace("nan", "NA")
    s["sexo"] = _col(raw, "SEXO").map({"1": "M", "3": "F"}).fillna("NA")
    s["cnes"] = _col(raw, "CNES")
    s["polo"] = _col(raw, "MUNIC_MOV") == POLO_MUNICIPIO
    return s


KEY_COLUMNS = ("NASC", "SEXO", "CEP", "MUNIC_RES")
# permanência (21 a 29): a AIH fecha e o paciente segue internado no mesmo hospital
PERMANENCIA = [str(c) for c in range(21, 30)]
TRANSFERENCIA = ["31"]  # transferência para outro hospital; a 32, para internação domiciliar, fica de fora


def link_episodes(raw: pd.DataFrame, max_gap_days: int = 1) -> pd.DataFrame:
    """Liga uma AIH de permanência (COBRANCA 21-29) e uma de transferência (COBRANCA 31) à AIH seguinte do mesmo
    paciente, dentro de max_gap_days da saída (mesmo hospital na permanência, outro hospital na transferência); o
    paciente é identificado por uma chave aproximada: data de nascimento, sexo, CEP e município de residência.
    Portela et al. (1997) juntavam as AIH de uma mesma internação usando, entre outros campos, o hospital, a data
    de nascimento e o sexo.

    Devolve, por AIH, o episódio, se é a primeira dele, o tipo de ligação e o óbito no episódio."""
    d = pd.DataFrame(index=raw.index)
    d["chave"] = raw[list(KEY_COLUMNS)].astype(str).agg("|".join, axis=1)
    d["ini"] = pd.to_datetime(raw["DT_INTER"].astype(str), format="%Y%m%d", errors="coerce")
    d["fim"] = pd.to_datetime(raw["DT_SAIDA"].astype(str), format="%Y%m%d", errors="coerce")
    d["cnes"] = raw["CNES"].astype(str)
    d["cob"] = raw["COBRANCA"].astype(str).str.strip()
    d["morte"] = (raw["MORTE"].astype(str).str.strip() == "1").astype(int)
    d = d.sort_values(["chave", "ini", "fim"], kind="stable")
    nxt = d.groupby("chave").shift(-1)
    gap = (nxt["ini"] - d["fim"]).dt.days
    close = gap.between(0, max_gap_days)
    cont = d["cob"].isin(PERMANENCIA) & (nxt["cnes"] == d["cnes"]) & close
    transf = d["cob"].isin(TRANSFERENCIA) & (nxt["cnes"] != d["cnes"]) & nxt["cnes"].notna() & close
    links_next = (cont | transf).fillna(False).astype(bool)
    linked_from_prev = links_next.groupby(d["chave"]).shift(1, fill_value=False).astype(bool)
    d["primeira_do_episodio"] = ~linked_from_prev
    d["episodio"] = d["primeira_do_episodio"].cumsum()
    d["ligacao"] = ""
    d.loc[cont.fillna(False).astype(bool), "ligacao"] = "continuacao"
    d.loc[transf.fillna(False).astype(bool), "ligacao"] = "transferencia"
    d["obito_episodio"] = d.groupby("episodio")["morte"].transform("max")
    return d.loc[raw.index, ["episodio", "primeira_do_episodio", "ligacao", "obito_episodio"]]


def link_rates(raw: pd.DataFrame, links: pd.DataFrame) -> dict:
    """Fração das AIH de permanência e de transferência que foram ligadas a uma AIH seguinte."""
    cob = raw["COBRANCA"].astype(str).str.strip()
    perm = cob.isin(PERMANENCIA)
    tr = cob.isin(TRANSFERENCIA)
    return {
        "permanencia_ligadas": float((links.loc[perm, "ligacao"] == "continuacao").mean())
        if perm.any()
        else float("nan"),
        "transferencias_ligadas": float((links.loc[tr, "ligacao"] == "transferencia").mean())
        if tr.any()
        else float("nan"),
    }
