import pandas as pd

from tests.dados_sinteticos import synthetic_raw
from triagem_baixo_risco.features import (
    LEAKY_COLUMNS,
    SOURCE_COLUMNS,
    age_in_years,
    build_features,
    build_strata,
    icd10_chapter,
    link_episodes,
    link_rates,
)


def test_age_in_years_converts_days_months_and_years():
    assert age_in_years(idade=30, cod_idade=2) == 0
    assert age_in_years(idade=18, cod_idade=3) == 1
    assert age_in_years(idade=45, cod_idade=4) == 45


def test_icd10_chapter_maps_code_ranges():
    assert icd10_chapter("I21") == "IX"
    assert icd10_chapter("J189") == "X"
    assert icd10_chapter("C50") == "II"
    assert icd10_chapter("O80") == "XV"
    assert icd10_chapter("S72") == "XIX"
    assert icd10_chapter("") is None


def _raw_row(**overrides):
    row = {
        "ANO_CMPT": "2024",
        "MES_CMPT": "01",
        "IDADE": "70",
        "COD_IDADE": "4",
        "SEXO": "3",
        "RACA_COR": "03",
        "MUNIC_RES": "314330",
        "MUNIC_MOV": "314330",
        "CNES": "2200000",
        "ESPEC": "01",
        "CAR_INT": "02",
        "DIAG_PRINC": "I219",
        "PROC_SOLIC": "0303060212",
        "COMPLEX": "03",
        "DT_INTER": "20240103",
        "DIAS_PERM": "12",
        "UTI_MES_TO": "3",
        "VAL_TOT": "5000.0",
        "MORTE": "1",
        "COBRANCA": "41",
        "PROC_REA": "0303060212",
    }
    row.update(overrides)
    return row


def test_build_features_uses_only_admission_time_information():
    raw = pd.DataFrame([_raw_row()])
    X, y = build_features(raw)
    assert set(X.columns).isdisjoint(LEAKY_COLUMNS)
    assert y["morte"].tolist() == [1]
    assert y["perm_prolongada"].tolist() == [1]
    assert X["idade_anos"].tolist() == [70]
    assert X["cid_capitulo"].tolist() == ["IX"]
    assert X["urgencia"].tolist() == [1]
    assert X["mesmo_municipio"].tolist() == [1]


def test_source_columns_never_include_post_admission_fields():
    assert set(SOURCE_COLUMNS).isdisjoint(LEAKY_COLUMNS)
    assert "COMPLEX" in LEAKY_COLUMNS
    assert "RACA_COR" not in SOURCE_COLUMNS


def test_build_features_drops_race_and_complexity():
    X, _ = build_features(pd.DataFrame([_raw_row()]))
    assert "raca_cor" not in X.columns and "complexidade" not in X.columns


def test_build_strata_flags_censored_obstetric_adult_and_age_band():
    raw = pd.DataFrame(
        [
            _raw_row(COBRANCA="27", IDADE="70"),
            _raw_row(COBRANCA="31", IDADE="30", DIAG_PRINC="O800", MORTE="0"),
            _raw_row(COBRANCA="12", IDADE="10", CAR_INT="01", MORTE="0", MUNIC_MOV="310120"),
            _raw_row(COBRANCA="20", IDADE="45", MORTE="0"),
        ]
    )
    s = build_strata(raw)
    assert s["censurada"].tolist() == [True, True, False, False]
    assert s["obstetrica"].tolist() == [False, True, False, False]
    assert s["adulto"].tolist() == [True, True, False, True]
    assert s["urgencia"].tolist() == [True, True, False, True]
    assert s["faixa_etaria"].tolist() == ["60-79", "18-39", "0-17", "40-59"]
    assert s["polo"].tolist() == [True, True, False, True]


def _aih(nasc, cnes, ini, fim, cob, morte="0", cep="39400000"):
    return {
        "NASC": nasc,
        "SEXO": "1",
        "CEP": cep,
        "MUNIC_RES": "314330",
        "CNES": cnes,
        "DT_INTER": ini,
        "DT_SAIDA": fim,
        "COBRANCA": cob,
        "MORTE": morte,
    }


def test_continuation_and_transfer_are_linked_and_death_is_carried_to_the_episode():
    raw = pd.DataFrame(
        [
            _aih("19500101", "H1", "20240301", "20240305", "27"),  # continua no mesmo hospital
            _aih("19500101", "H1", "20240305", "20240320", "41", morte="1"),
            _aih("19600101", "H1", "20240401", "20240410", "31"),  # transferido
            _aih("19600101", "H2", "20240411", "20240420", "12"),
            _aih("19700101", "H1", "20240501", "20240505", "31"),  # transferência sem AIH seguinte próxima
            _aih("19700101", "H2", "20240512", "20240515", "12"),
            _aih("19800101", "H1", "20240601", "20240605", "26"),  # permanência, mas a seguinte é em outro hospital
            _aih("19800101", "H2", "20240605", "20240610", "12"),
        ]
    )
    out = link_episodes(raw)
    assert out["primeira_do_episodio"].tolist() == [True, False, True, False, True, True, True, True]
    assert out["ligacao"].tolist() == ["continuacao", "", "transferencia", "", "", "", "", ""]
    assert out["obito_episodio"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert out.loc[0, "episodio"] == out.loc[1, "episodio"] != out.loc[2, "episodio"]


def test_different_patients_are_never_linked():
    raw = pd.DataFrame(
        [
            _aih("19500101", "H1", "20240301", "20240305", "27"),
            _aih("19500101", "H1", "20240305", "20240310", "12", cep="39400999"),
        ]
    )
    assert link_episodes(raw)["primeira_do_episodio"].tolist() == [True, True]


def test_link_rates_report_share_of_linked_closures():
    raw = pd.DataFrame(
        [
            _aih("19500101", "H1", "20240301", "20240305", "27"),
            _aih("19500101", "H1", "20240305", "20240310", "12"),
            _aih("19600101", "H1", "20240401", "20240410", "31"),
            _aih("19700101", "H1", "20240501", "20240505", "20"),  # "20" não é motivo de permanência
            _aih("19700101", "H1", "20240505", "20240510", "12"),
            _aih("19800101", "H1", "20240601", "20240605", "32"),  # "32" é transferência para internação domiciliar
            _aih("19800101", "H2", "20240605", "20240610", "12"),
        ]
    )
    links = link_episodes(raw)
    assert links["ligacao"].tolist()[3] == "" and links["ligacao"].tolist()[5] == ""
    assert link_rates(raw, links) == {"permanencia_ligadas": 1.0, "transferencias_ligadas": 0.0}
