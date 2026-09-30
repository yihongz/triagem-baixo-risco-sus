import json

import pytest

from tests.dados_sinteticos import synthetic_raw
from triagem_baixo_risco import config
from triagem_baixo_risco.experimentos.exploratorios import _json_default, prepare, run_x3, run_x4, run_x5, run_x6
from triagem_baixo_risco.plots import LABELS, STYLES, make_all, panel_titles


def test_make_all_draws_every_figure_from_saved_results(tmp_path):
    d = prepare(synthetic_raw(n=20000, seed=9, logit_shift={2020: 0.5, 2021: 0.8}))
    results = {
        "x3": run_x3(d, strategies=("a", "c")),
        "x4": run_x4(d, strategies=("a", "c")),
        "x5": run_x5(d),
        "x6": run_x6(d, alphas=(0.05,), n_boot=20, n_importance=2000, n_boot_curve=10),
    }
    for name, res in results.items():
        (tmp_path / f"resultados_{name}.json").write_text(json.dumps(res, default=_json_default))
    made = make_all(tmp_path, out_dir=tmp_path / "figuras")
    assert {p.name for p in made} == {
        "garantias_x3.png",
        "garantias_x4.png",
        "equidade_x5.png",
        "calibracao_x6.png",
        "utilidade_x6.png",
    }
    assert all(p.stat().st_size > 10_000 for p in made)


def test_panel_titles_follow_the_outcome():
    morte, perm = panel_titles("morte"), panel_titles("perm_prolongada")
    assert morte[0] != perm[0] and morte[1] != perm[1]
    assert not any("óbito" in title for title in perm)


def test_styles_are_distinct_and_cover_the_strategies():
    assert set(STYLES) == set(LABELS) == set("abcdefg")
    assert len({(style["color"], style["marker"]) for style in STYLES.values()}) == 7


def test_make_all_refuses_a_folder_without_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_all(tmp_path / "nada")
    assert not (tmp_path / "nada").exists()
    with pytest.raises(FileNotFoundError):
        make_all(tmp_path)  # a pasta existe, mas sem resultados_*.json


def test_make_all_uses_the_figures_folder_of_the_outcome_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "fig")
    d = prepare(synthetic_raw(n=20000, seed=9, logit_shift={2020: 0.5, 2021: 0.8}))
    results = {
        "x3": run_x3(d, strategies=("a", "c")),
        "x4": run_x4(d, strategies=("a", "c")),
        "x5": run_x5(d),
        "x6": run_x6(d, alphas=(0.05,), n_boot=20, n_importance=2000, n_boot_curve=10),
    }
    for name, res in results.items():
        (tmp_path / f"resultados_{name}.json").write_text(json.dumps(res, default=_json_default))
    (tmp_path / "resumo.json").write_text(json.dumps({"desfecho": "perm_prolongada"}))
    made = make_all(tmp_path)
    expected_dir = tmp_path / "fig" / "perm_prolongada"
    assert made
    assert all(p.parent == expected_dir for p in made)
    assert {p.name for p in made} == {
        "garantias_x3.png",
        "garantias_x4.png",
        "equidade_x5.png",
        "calibracao_x6.png",
        "utilidade_x6.png",
    }
