import json

import pandas as pd
import pytest

from tests.dados_sinteticos import synthetic_raw
from triagem_baixo_risco.dataset import SealedDataError, admission_dates, in_window, sealed_path, split_sealed
from triagem_baixo_risco.experimentos.confirmatorio import (
    VARIANTS,
    check_followup_completeness,
    competence_filter,
    h2_conditional,
    main,
    run_confirmatory,
)


def _write(directory, open_df, sealed_df):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sih.parquet"
    open_df.to_parquet(path)
    sealed_df.to_parquet(sealed_path(path))
    return path


@pytest.fixture(scope="module")
def raw():
    # competência até 4 meses depois da internação: meses contínuos até dez/2026, cobrindo os cortes de jul e out
    return synthetic_raw(n=26000, seed=11, years=(2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026), cmpt_delay_max=4)


@pytest.fixture(scope="module")
def path(raw, tmp_path_factory):
    return _write(tmp_path_factory.mktemp("x7"), *split_sealed(raw))


@pytest.fixture(scope="module")
def x7a(path, tmp_path_factory):
    """Uma execução de X7a, copiada pelos testes que precisam de um resultado anterior."""
    out = tmp_path_factory.mktemp("confirmatorio")
    return run_confirmatory(path, "a", out, n_boot=50), out / "x7a.json"


def test_confirmatory_run_evaluates_the_preregistered_hypotheses(raw, x7a):
    res, file = x7a
    r = res["alfas"]["0.05"]
    assert {"regra", "h1", "h1b", "h2", "h4", "h5"} <= set(r)
    assert res["n_teste"] == int(in_window(admission_dates(raw), *VARIANTS["a"]["test"]).sum())
    assert len(res["sha256_dados"]["lacrado"]) == 64 and res["arquivo"] == "x7a.json"
    assert res["escopo_h5"] == "descritiva" and res["sensibilidade"] == "nenhuma"
    comp = res["competencias"]
    assert (comp["primeira"], comp["ultima"]) == ("2019-01", "2026-07") and comp["linhas_fora_do_corte"] > 0
    assert res["integridade"]["ident"] == "todas 1" and res["info_preparo"] == {"descartadas_sem_data": 0}
    assert {"python", "numpy", "pandas", "scipy", "sklearn", "pacote"} <= set(res["ambiente"])
    assert r["regra"]["hgb"]["n1"] > 0 and {"0-17", "80+"} <= set(r["regra"]["cruzado"]["n1"])
    assert r["metricas"]["g1_nulo"] is not None and "agregado_descritivo" in r["h5"]["faixas"]["80+"]
    assert json.loads(file.read_text())["variante"] == "a"


def test_x7c_needs_x7a_and_aborts_unless_the_refitted_rule_is_identical(raw, path, x7a, tmp_path):
    out = tmp_path / "confirmatorio"
    with pytest.raises(RuntimeError, match="rode X7a antes"):
        run_confirmatory(path, "c", out, n_boot=50)
    out.mkdir()
    (out / "x7a.json").write_bytes(x7a[1].read_bytes())
    res = run_confirmatory(path, "c", out, n_boot=50)
    assert res["regra_de_referencia"] == "x7a.json" and res["escopo_h5"] == "confirmatoria"
    assert res["competencias"]["ultima"] == "2026-10"
    assert res["n_teste"] == int(in_window(admission_dates(raw), *VARIANTS["c"]["test"]).sum())
    ref = json.loads((out / "x7a.json").read_text())
    ref["alfas"]["0.05"]["regra"]["hgb"]["q1"] = 0.123456
    (out / "x7a.json").write_text(json.dumps(ref))
    with pytest.raises(RuntimeError, match="hgb.q1"):
        run_confirmatory(path, "c", out, n_boot=50, deviation="teste da regra congelada")
    assert [p.name for p in out.glob("x7c*.json")] == ["x7c.json"] and not (out / "desvios.md").exists()


def test_competence_filter_checks_coverage_on_every_row_and_then_drops_rows_billed_after_the_cutoff():
    fit = pd.period_range("2019-01", "2026-07", freq="M")
    rows = [(f"{m.year}{m.month:02d}15", m.year, m.month) for m in fit]  # cobradas no mês da internação
    rows += [("20260615", 2026, 9), ("20260615", 2026, 11)]  # teste de X7c: até out/2026
    rows += [("20260710", 2026, 8), ("20260805", 2026, 9), ("20260910", 2026, 10)]  # fora de toda janela de X7c
    df = pd.DataFrame(rows, columns=["DT_INTER", "ANO_CMPT", "MES_CMPT"]).astype(str)
    keep, info = competence_filter(df, VARIANTS["c"])  # ago e out/2026 só em linhas fora das janelas
    assert keep.tolist() == [True] * len(fit) + [True, False] + [False] * 3
    assert info["linhas_fora_do_corte"] == 4 and (info["primeira"], info["ultima"]) == ("2019-01", "2026-09")
    assert "2026-08" not in info["linhas_por_competencia"]
    with pytest.raises(ValueError, match="2020-05"):
        competence_filter(df[~((df["ANO_CMPT"] == "2020") & (df["MES_CMPT"] == "5"))], VARIANTS["c"])
    with pytest.raises(ValueError, match="2026-10"):
        competence_filter(df.iloc[:-1], VARIANTS["c"])  # out/2026 não entregue pelo ETL


def test_x7c_passes_coverage_when_the_late_months_exist_only_outside_every_window(raw, x7a, tmp_path):
    dates = admission_dates(raw)
    test_c = in_window(dates, *VARIANTS["c"]["test"])
    moved = raw.copy()  # teste de X7c cobrado no mês da internação
    moved.loc[test_c, "ANO_CMPT"] = dates[test_c].dt.year.astype(str)
    moved.loc[test_c, "MES_CMPT"] = dates[test_c].dt.month.map("{:02d}".format)
    cmpt = moved["ANO_CMPT"].astype(int) * 100 + moved["MES_CMPT"].astype(int)
    late = cmpt.between(202608, 202610).to_numpy()
    assert late.any() and (dates[late] >= "2026-07-01").all()  # ago-out/2026 só em internações de jul/2026 em diante
    out = tmp_path / "confirmatorio"
    out.mkdir()
    (out / "x7a.json").write_bytes(x7a[1].read_bytes())
    res = run_confirmatory(_write(tmp_path / "dados", *split_sealed(moved)), "c", out, n_boot=50)
    assert res["regra_de_referencia"] == "x7a.json" and res["competencias"]["ultima"] == "2026-07"


def test_confirmatory_run_refuses_data_with_a_missing_competence_month(raw, tmp_path):
    gap = raw[~((raw["ANO_CMPT"] == "2020") & (raw["MES_CMPT"] == "05"))]
    path = _write(tmp_path / "dados", *split_sealed(gap))
    with pytest.raises(ValueError, match="2020-05"):
        run_confirmatory(path, "a", tmp_path / "out")


def test_deviation_writes_a_new_file_and_keeps_the_first_result(path, x7a, tmp_path):
    (tmp_path / "x7a.json").write_bytes(x7a[1].read_bytes())
    before = (tmp_path / "x7a.json").read_bytes()
    with pytest.raises(RuntimeError, match="já foi executada"):
        run_confirmatory(path, "a", tmp_path, n_boot=50)
    res = run_confirmatory(path, "a", tmp_path, n_boot=50, deviation="teste de reexecução")
    assert res["arquivo"].startswith("x7a_desvio_") and (tmp_path / res["arquivo"]).exists()
    assert (tmp_path / "x7a.json").read_bytes() == before
    assert f"X7a: teste de reexecução (arquivo {res['arquivo']})" in (tmp_path / "desvios.md").read_text()


def test_sensitivities_run_only_on_x7a_and_write_their_own_file(path, x7a, tmp_path):
    res = run_confirmatory(path, "a", tmp_path, n_boot=50, sensitivity="s3")
    assert res["arquivo"] == "x7a_s3.json" and (tmp_path / "x7a_s3.json").exists()
    assert res["sensibilidade"] == "s3" and res["n_teste"] < x7a[0]["n_teste"]  # S3 tira as censuradas
    with pytest.raises(ValueError, match="só em X7a"):
        run_confirmatory(path, "b", tmp_path, n_boot=50, sensitivity="s1")


def test_confirmatory_run_checks_the_integrity_of_the_data(raw, tmp_path):
    open_df, sealed_df = split_sealed(raw)
    dup = sealed_df.copy()
    dup["N_AIH"] = dup["N_AIH"].where(dup.index != dup.index[0], open_df["N_AIH"].iloc[0])
    with pytest.raises(ValueError, match="1 N_AIH"):
        run_confirmatory(_write(tmp_path / "dup", open_df, dup), "a", tmp_path / "out")
    early = sealed_df.copy()
    early["DT_INTER"] = early["DT_INTER"].where(early.index != early.index[0], "20251231")
    with pytest.raises(SealedDataError):
        run_confirmatory(_write(tmp_path / "early", open_df, early), "a", tmp_path / "out")
    ident = sealed_df.copy()
    ident["IDENT"] = ident["IDENT"].where(ident.index != ident.index[0], "5")
    with pytest.raises(ValueError, match="IDENT"):
        run_confirmatory(_write(tmp_path / "ident", open_df, ident), "a", tmp_path / "out")


def test_cli_prints_h5_only_where_it_is_evaluated_and_the_actual_file(path, tmp_path, capsys, monkeypatch):
    import triagem_baixo_risco.experimentos.confirmatorio

    real = triagem_baixo_risco.experimentos.confirmatorio.run_confirmatory
    monkeypatch.setattr(
        triagem_baixo_risco.experimentos.confirmatorio, "run_confirmatory", lambda *a, **k: real(*a, **k, n_boot=50)
    )
    main(["--data", str(path), "--variante", "b", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert "H1 refutada=" in printed and "H5" not in printed and f"resultados em {tmp_path / 'x7b.json'}" in printed


def test_h2_conditional_prediction_depends_on_the_prevalence_ratio():
    assert h2_conditional({"razao_prevalencia": 0.9, "p_g2": 0.5})["refutada"] is False
    assert h2_conditional({"razao_prevalencia": 0.9, "p_g2": 0.01})["refutada"] is True
    assert h2_conditional({"razao_prevalencia": 1.3, "p_g2": 0.5})["refutada"] is True
    assert h2_conditional({"razao_prevalencia": 1.1, "p_g2": 0.01})["refutada"] is None


def test_confirmatory_run_explains_a_missing_sealed_file(tmp_path):
    raw = synthetic_raw(n=500, seed=12)
    raw.to_parquet(tmp_path / "sih.parquet")
    with pytest.raises(FileNotFoundError, match="lacrado"):
        run_confirmatory(tmp_path / "sih.parquet", "a", tmp_path / "out")


def _followup_frame(rows):
    """DataFrame mínimo para a checagem de seguimento: lista de (DT_INTER, n)."""
    return pd.DataFrame([{"DT_INTER": date} for date, n in rows for _ in range(n)])


def test_followup_check_passes_when_test_months_match_previous_year():
    raw = _followup_frame([("20250115", 100), ("20250215", 100), ("20260115", 95), ("20260215", 101)])
    ratios = check_followup_completeness(raw, "2026-01-01", "2026-03-01")
    assert ratios == {"2026-01": pytest.approx(0.95), "2026-02": pytest.approx(1.01)}


def test_followup_check_refuses_incomplete_month():
    raw = _followup_frame([("20250115", 100), ("20250215", 100), ("20260115", 95), ("20260215", 80)])
    with pytest.raises(ValueError, match=r"2026-02.*0\.80"):
        check_followup_completeness(raw, "2026-01-01", "2026-03-01")
