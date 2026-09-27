import json

import pytest

from tests.dados_sinteticos import synthetic_raw
from triagem_baixo_risco.dataset import in_window
from triagem_baixo_risco.experimentos.exploratorios import (
    SENSITIVITIES,
    exploratory_hypotheses,
    fit_baselines,
    main,
    prepare,
    run_x1,
    run_x2,
    run_x3,
    run_x4,
    run_x5,
    run_x6,
    run_x8,
    strategy_summary,
)
from triagem_baixo_risco.modeling.train import CATEGORICAL, NUMERIC


@pytest.fixture(scope="module")
def raw():
    return synthetic_raw(n=24000, seed=7, logit_shift={2020: 0.5, 2021: 0.8})


@pytest.fixture(scope="module")
def data(raw):
    return prepare(raw)


def test_prepare_supports_every_sensitivity(raw):
    for s in SENSITIVITIES:
        d = prepare(raw, "morte", s)
        assert len(d.X) == len(d.y) == len(d.dates) == len(d.strata) > 0
    assert "cid3" not in prepare(raw, "morte", "s2").X.columns
    assert "cnes" not in prepare(raw, "morte", "s6").X.columns
    assert len(prepare(raw, "morte", "s3").y) < len(raw)
    assert "taxas_ligacao" in prepare(raw, "morte", "s5").info
    with pytest.raises(ValueError):
        prepare(raw, "perm_prolongada", "s5")


def test_x2_reports_twelve_quarters_with_both_guarantees(data):
    rows = run_x2(data)
    assert [r["trimestre"] for r in rows][0] == "2023T1" and len(rows) == 12
    assert {"fdl", "p_g1", "tol", "r_hat", "p_g2", "oe", "razao_prevalencia"} <= set(rows[0])


def test_x3_and_x4_run_every_strategy(data):
    x3 = run_x3(data)
    variants = {"e(gamma=0.01)", "e(gamma=0.05)", "f(meia_vida=2)", "f(meia_vida=8)"}
    assert set(x3) == {"a", "b", "c", "d", "e", "f", "g"} | variants and all(len(v) == 12 for v in x3.values())
    x4 = run_x4(data)
    assert set(x4) == {"a", "b", "c", "e", "f", "g"} | variants and all(len(v) == 8 for v in x4.values())
    assert not variants & set(exploratory_hypotheses({"x3": x3})["h1"]["cenarios"])
    assert x4["a"][5]["razao_prevalencia"] > 1.0  # 2021 com risco maior que out-dez/2019


def test_x5_compares_global_and_cross_rules_by_age_band(data):
    out = run_x5(data)
    bands = {r["nivel"] for r in out["cruzado"] if r["variavel"] == "faixa_etaria"}
    assert {"0-17", "80+"} <= bands and set(out) == {"global", "cruzado"}


def test_x6_compares_the_model_with_both_baselines(data):
    out = run_x6(data, alphas=(0.1,), n_boot=50, n_importance=3000, n_boot_curve=10)
    res = out["alfas"]["0.1"]
    assert {"hgb", "logistica", "logistica_simples", "regra_sem_modelo", "diferencas", "banda_custo"} <= set(res)
    for k in ("regra", "logistica_simples"):  # IC de hospitais em "ic" (H4) e de internações ao lado
        assert len(res["diferencas"][k]["ic"]) == 2 and len(res["diferencas"][k]["ic_linhas"]) == 2
        assert res["diferencas"][k]["ic"] != res["diferencas"][k]["ic_linhas"]
    assert "ici" in res["metricas_hgb"] and len(res["curva_calibracao"]["p"]) > 10
    assert "banda_inferior" in res["curva_calibracao"] and len(res["bootstrap_hgb"]["auroc"]["hospitais"]) == 2
    assert out["importancia_blocos"][0]["bloco"] in {"idade", "cid", "carater"}


def test_x1_and_x8_describe_and_monitor_drift(raw, data):
    table, domain = run_x1(raw, data)
    assert "2020T1" in table.index and set(domain) == {2020, 2021, 2022, 2023, 2024, 2025}
    mon = run_x8(data)
    assert set(mon) == {"x4", "x2"} and {"obitos", "sobreviventes", "oe"} <= set(mon["x4"])


def test_exploratory_hypotheses_combine_the_experiments(data):
    results = {"x3": run_x3(data, strategies=("a", "c")), "x4": run_x4(data, strategies=("a", "c")), "x5": run_x5(data)}
    out = exploratory_hypotheses(results)
    assert {"h1", "h1b", "h2", "h3", "h5"} <= set(out)
    assert out["h3"]["n_trimestres"] == 20


def test_cli_accumulates_experiments_run_separately(tmp_path, raw):
    path = tmp_path / "sih.parquet"
    raw.to_parquet(path)
    out = tmp_path / "out" / "morte" / "nenhuma"
    main(["--data", str(path), "--experimento", "x2", "--out", str(tmp_path / "out")])
    main(["--data", str(path), "--experimento", "x5", "--out", str(tmp_path / "out")])
    summary = json.loads((out / "resumo.json").read_text())
    assert summary["experimentos_disponiveis"] == ["x2", "x5"]
    assert {"h1", "h5"} <= set(summary["hipoteses_exploratorias"])
    assert (out / "x2.csv").exists() and (out / "x5.csv").exists()


def test_prepare_counts_rows_dropped_for_invalid_admission_date(raw):
    bad = raw.head(50).copy()
    bad.loc[bad.index[:3], "DT_INTER"] = "00000000"
    assert prepare(bad).info["descartadas_sem_data"] == 3


def test_rows_without_a_valid_admission_date_pass_load_open_and_are_counted_by_prepare(raw, tmp_path):
    from triagem_baixo_risco.dataset import load_open

    bad = raw.head(200).copy()
    bad.loc[bad.index[:3], "DT_INTER"] = ["00000000", "", "20231345"]
    bad.to_parquet(tmp_path / "sih.parquet")
    assert prepare(load_open(tmp_path / "sih.parquet")).info["descartadas_sem_data"] == 3


def test_exploratory_code_never_references_the_sealed_loader():
    import inspect

    import triagem_baixo_risco.experimentos.exploratorios

    assert "load_sealed" not in inspect.getsource(triagem_baixo_risco.experimentos.exploratorios)


def test_strategy_summary_reports_calibration_drift_and_violations(data):
    out = strategy_summary({"x3": run_x3(data, strategies=("a", "c"))})
    assert {"x3_a", "x3_c"} <= set(out)
    assert {"oe_desvio_medio", "trimestres_g1_violada", "trimestres_g2_violada", "fracao_liberada_media"} <= set(
        out["x3_a"]
    )


def test_fit_baselines_follows_the_sensitivity_columns(raw):
    """Nas sensibilidades S2 e S6, a logística completa usa as colunas da própria sensibilidade
    (d.numeric/d.categorical), como o boosting já fazia, em vez das listas fixas NUMERIC/CATEGORICAL."""
    removed = {"s2": {"cid3"}, "s6": {"cnes", "munic_mov"}}
    for s in ("s2", "s6"):
        d = prepare(raw, "morte", s)
        train = in_window(d.dates, "2019-01-01", "2022-01-01")
        models, preds = fit_baselines(d, train)
        for p in preds.values():
            assert len(p) == len(d.y)
            assert ((p >= 0) & (p <= 1)).all()
        cat_cols = set(models["logistica"].named_steps["pre"].transformers[-1][2])
        assert not removed[s] & cat_cols


def test_full_logistic_lists_unchanged_outside_s2_s6(raw):
    """Fora de S2 e S6, fit_baselines monta a logística completa com as listas fixas NUMERIC e CATEGORICAL."""
    for s in ("nenhuma", "s1", "s3", "s4", "s5"):
        d = prepare(raw, "morte", s)
        assert d.numeric == list(NUMERIC)
        assert d.categorical == list(CATEGORICAL)
