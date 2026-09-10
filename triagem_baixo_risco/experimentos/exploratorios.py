"""Experimentos exploratórios X1 a X6 e X8, com as sensibilidades S1 a S6.

Uso:
    python -m triagem_baixo_risco.experimentos.exploratorios --experimento todos --desfecho morte --sensibilidade nenhuma

Usa só internações anteriores a 2026. Saídas em reports/exploratorios/<desfecho>/<sensibilidade>/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from triagem_baixo_risco import config
from triagem_baixo_risco.avaliacao import (
    bootstrap_ci,
    bootstrap_metrics,
    calibration_curve_points,
    evaluate_period,
    evaluate_strategy_rows,
    h1,
    h1b,
    h2,
    h3,
    h5,
    integrated_calibration_index,
)
from triagem_baixo_risco.dataset import admission_dates, in_window, load_open, quarter_labels
from triagem_baixo_risco.features import build_features, build_strata, link_episodes, link_rates
from triagem_baixo_risco.modeling.atualizacao import STRATEGIES, quarter_starts, run_strategy
from triagem_baixo_risco.modeling.predict import apply_rule, choose_band, death_scores, fraction_deaths_released, holm
from triagem_baixo_risco.modeling.train import (
    CATEGORICAL,
    FEATURE_BLOCKS,
    NUMERIC,
    block_permutation_importance,
    fit_window,
    make_hgb,
    make_logistic_full,
    make_logistic_simple,
    model_frame,
    rule_based_release,
)
from triagem_baixo_risco.monitoramento import conformal_pvalues, first_alarm, mixture_power_martingale_log10, oe_control

SENSITIVITIES = ("nenhuma", "s1", "s2", "s3", "s4", "s5", "s6")
ALPHA = 0.05
SEED = config.SEED
X3_SETUP = {
    "train": ("2019-01-01", "2022-01-01"),
    "base": ("2022-01-01", "2023-01-01"),
    "quarters": ("2023-01-01", "2025-10-01"),
}
X4_SETUP = {
    "train": ("2019-01-01", "2019-10-01"),
    "base": ("2019-10-01", "2020-01-01"),
    "quarters": ("2020-01-01", "2021-10-01"),
}
X4_STRATEGIES = ("a", "b", "c", "e", "f", "g")
PARAMETER_VARIANTS = {
    "e(gamma=0.01)": ("e", {"gamma": 0.01}),
    "e(gamma=0.05)": ("e", {"gamma": 0.05}),
    "f(meia_vida=2)": ("f", {"half_life": 2.0}),
    "f(meia_vida=8)": ("f", {"half_life": 8.0}),
}


@dataclass
class Data:
    X: pd.DataFrame
    y: np.ndarray
    dates: pd.Series
    strata: pd.DataFrame
    numeric: list
    categorical: list
    codes: np.ndarray = None  # CID de 3 caracteres (para H1b), mesmo quando cid3 sai do modelo
    calibrator: str = "platt"
    info: dict = field(default_factory=dict)


def prepare(raw: pd.DataFrame, target: str = "morte", sensitivity: str = "nenhuma") -> Data:
    """Matriz do modelo, desfecho, datas de internação e estratos, com a sensibilidade pedida aplicada."""
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"sensibilidade desconhecida: {sensitivity}")
    raw = raw.reset_index(drop=True)
    X, Y = build_features(raw)
    strata = build_strata(raw)
    y = Y[target].to_numpy().astype(int)
    dates = admission_dates(raw)
    numeric, categorical, calibrator, info = list(NUMERIC), list(CATEGORICAL), "platt", {}
    keep = np.ones(len(raw), dtype=bool)
    if sensitivity == "s1":
        calibrator = "isotonic"
    elif sensitivity == "s2":
        categorical.remove("cid3")
    elif sensitivity == "s3":
        keep = ~strata["censurada"].to_numpy()
    elif sensitivity == "s4":
        dates = pd.to_datetime(
            raw["ANO_CMPT"].astype(str) + raw["MES_CMPT"].astype(str).str.zfill(2) + "01",
            format="%Y%m%d",
            errors="coerce",
        )
    elif sensitivity == "s5":
        if target != "morte":
            raise ValueError("S5 (desfecho por episódio) só se aplica ao óbito")
        links = link_episodes(raw)
        info["taxas_ligacao"] = link_rates(raw, links)
        keep = links["primeira_do_episodio"].to_numpy()
        y = links["obito_episodio"].to_numpy().astype(int)
    elif sensitivity == "s6":
        categorical = [c for c in categorical if c not in ("cnes", "munic_mov")]
    info["descartadas_sem_data"] = int((keep & dates.isna().to_numpy()).sum())
    keep &= dates.notna().to_numpy()
    idx = np.flatnonzero(keep)
    Xm = model_frame(X, numeric, categorical)
    return Data(
        Xm.iloc[idx].reset_index(drop=True),
        y[idx],
        dates.iloc[idx].reset_index(drop=True),
        strata.iloc[idx].reset_index(drop=True),
        numeric,
        categorical,
        X["cid3"].astype(str).to_numpy()[idx],
        calibrator,
        info,
    )


def _predict(pipe, X: pd.DataFrame, chunk: int = 200_000) -> np.ndarray:
    return np.concatenate([pipe.predict_proba(X.iloc[a : a + chunk])[:, 1] for a in range(0, len(X), chunk)])


def predict_raw(d: Data, train_mask, seed: int = SEED) -> np.ndarray:
    """Treina o boosting nas linhas de treino e devolve as probabilidades brutas de todas as linhas."""
    pipe = make_hgb(seed, d.numeric, d.categorical).fit(d.X[train_mask], d.y[train_mask])
    return _predict(pipe, d.X)


def _strategy_experiment(d: Data, setup: dict, strategies, alpha: float) -> dict[str, list[dict]]:
    train = in_window(d.dates, *setup["train"])
    base = in_window(d.dates, *setup["base"])
    p_raw = predict_raw(d, train)
    refit = lambda cutoff: predict_raw(d, in_window(d.dates, setup["train"][0], cutoff))
    runs = [(s, s, {}) for s in strategies]
    runs += [
        (name, s, kw) for name, (s, kw) in PARAMETER_VARIANTS.items() if s in strategies
    ]  # sensibilidade de γ e meia-vida
    out = {}
    for name, s, kw in runs:
        rows = run_strategy(
            s,
            p_raw=p_raw,
            y=d.y,
            dates=d.dates,
            quarters=quarter_starts(*setup["quarters"]),
            base_mask=base,
            train_mask=train,
            kind=d.calibrator,
            alpha=alpha,
            refit=refit if s == "d" else None,
            **kw,
        )
        out[name] = evaluate_strategy_rows(rows, d.y, alpha, d.codes, d.dates)
    return out


def run_x2(d: Data, alpha: float = ALPHA) -> list[dict]:
    """Envelhecimento: treino até 2021, calibração única em 2022, trimestres de 2023 a 2025."""
    return _strategy_experiment(d, X3_SETUP, ("a",), alpha)["a"]


def run_x3(d: Data, alpha: float = ALPHA, strategies=STRATEGIES) -> dict[str, list[dict]]:
    return _strategy_experiment(d, X3_SETUP, strategies, alpha)


def run_x4(d: Data, alpha: float = ALPHA, strategies=X4_STRATEGIES) -> dict[str, list[dict]]:
    """Pandemia: treino jan-set/2019, calibração out-dez/2019, trimestres de 2020 e 2021."""
    return _strategy_experiment(d, X4_SETUP, strategies, alpha)


def run_x5(d: Data, alpha: float = ALPHA) -> dict[str, list[dict]]:
    """Equidade: Mondrian por classe (global) contra Mondrian cruzado, com recalibração anual."""
    p_raw = predict_raw(d, in_window(d.dates, "2019-01-01", "2022-01-01"))
    faixa = d.strata["faixa_etaria"].astype(str).to_numpy()
    out = {"global": [], "cruzado": []}
    for year in (2023, 2024, 2025):
        cal = in_window(d.dates, f"{year - 1}-01-01", f"{year}-01-01")
        test = in_window(d.dates, f"{year}-01-01", f"{year + 1}-01-01")
        for name, groups in (("global", None), ("cruzado", faixa)):
            f = fit_window(p_raw, d.y, cal, kind=d.calibrator, alpha=alpha, groups=groups)
            p_t = f.calibrate(p_raw[test])
            rel = apply_rule(f.rule, p_t, None if groups is None else groups[test])
            for var in ("faixa_etaria", "sexo", "polo"):
                levels = d.strata[var].astype(str).to_numpy()[test]
                for lv in np.unique(levels):
                    m = levels == lv
                    r = evaluate_period(
                        d.y[test][m],
                        p_t[m],
                        rel[m],
                        alpha,
                        f.rule.r_hat,
                        f.prevalence,
                        n1=f.rule.n1,
                        groups=None if groups is None else groups[test][m],
                    )
                    out[name].append({"ano": year, "variavel": var, "nivel": lv, **r})
    return out


def utility_comparisons(
    d: Data, preds: dict, cal, test, alpha: float, n_boot: int = 1000, n_boot_curve: int = 100
) -> dict:
    """Frações liberadas (todas e adultos não obstétricos) e diferenças contra as linhas de base, com IC por
    bootstrap de hospitais ("ic") e de internações ("ic_linhas"). A regra sem modelo é comparada ao boosting com o
    limiar que reproduz, na parte B, a FDL dela."""
    adult = (d.strata["adulto"] & ~d.strata["obstetrica"]).to_numpy()[test]
    yt = d.y[test]
    cnes_test = d.strata["cnes"].to_numpy()[test]
    fitted = {k: fit_window(p, d.y, cal, kind=d.calibrator, alpha=alpha) for k, p in preds.items()}
    p_test = {k: f.calibrate(preds[k][test]) for k, f in fitted.items()}
    rel = {k: apply_rule(f.rule, p_test[k]) for k, f in fitted.items()}
    res = {
        k: {
            "fracao_liberada": float(r.mean()),
            "fracao_liberada_adultos": float(r[adult].mean()),
            "fdl": fraction_deaths_released(yt, r)["value"],
        }
        for k, r in rel.items()
    }
    hgb = fitted["hgb"]
    ib = hgb.index_B
    rule_b = rule_based_release(d.strata.iloc[ib])
    fdl_rule_b = fraction_deaths_released(d.y[ib], rule_b)["value"]
    q_match = float(np.quantile(death_scores(hgb.p_B)[d.y[ib] == 1], 1 - fdl_rule_b))
    rel_match = death_scores(p_test["hgb"]) > q_match
    rule_t = rule_based_release(d.strata.iloc[np.flatnonzero(test)])
    res["regra_sem_modelo"] = {
        "fracao_liberada": float(rule_t.mean()),
        "fdl": fraction_deaths_released(yt, rule_t)["value"],
        "hgb_mesma_fdl_fracao": float(rel_match.mean()),
        "hgb_mesma_fdl_fdl": fraction_deaths_released(yt, rel_match)["value"],
    }

    def diff(a, b):
        stat = lambda i: a[i].mean() - b[i].mean()
        return {
            "diferenca": float(a.mean() - b.mean()),
            "ic": bootstrap_ci(stat, len(a), n_boot, SEED, clusters=cnes_test),
            "ic_linhas": bootstrap_ci(stat, len(a), n_boot, SEED),
        }

    res["diferencas"] = {
        "regra": diff(rel_match, rule_t),
        "logistica_simples": diff(rel["hgb"], rel["logistica_simples"]),
    }
    grid = tuple(float(t) for t in np.round(np.arange(0.005, 1.0, 0.005), 3))
    t_lo, _ = choose_band(d.y[ib], hgb.p_B, grid, c_fn=10.0, c_fp=1.0, c_defer=0.3)
    res["banda_custo"] = {"t_lo": t_lo, "limiar_conformal": float(1 - hgb.rule.q1)}
    res["metricas_hgb"] = evaluate_period(
        yt, p_test["hgb"], rel["hgb"], alpha, hgb.rule.r_hat, hgb.prevalence, n1=hgb.rule.n1
    )
    res["metricas_hgb"]["ici"] = integrated_calibration_index(yt, p_test["hgb"])
    res["curva_calibracao"] = calibration_curve_points(yt, p_test["hgb"], n_boot=n_boot_curve)
    res["bootstrap_hgb"] = bootstrap_metrics(yt, p_test["hgb"], rel["hgb"], cnes_test, n_boot)
    return res


def fit_baselines(d: Data, train) -> tuple[dict, dict]:
    models = {
        "hgb": make_hgb(SEED, d.numeric, d.categorical),
        "logistica": make_logistic_full(NUMERIC, CATEGORICAL),
        "logistica_simples": make_logistic_simple(),
    }
    preds = {k: _predict(m.fit(d.X[train], d.y[train]), d.X) for k, m in models.items()}
    return models, preds


def run_x6(
    d: Data, alphas=(0.05, 0.10), n_boot: int = 1000, n_importance: int = 30_000, n_boot_curve: int = 100
) -> dict:
    """Utilidade (treino 2019-2023, calibração 2024, teste 2025) e importância por permutação."""
    train = in_window(d.dates, "2019-01-01", "2024-01-01")
    cal = in_window(d.dates, "2024-01-01", "2025-01-01")
    test = in_window(d.dates, "2025-01-01", "2026-01-01")
    models, preds = fit_baselines(d, train)
    out = {"alfas": {str(a): utility_comparisons(d, preds, cal, test, a, n_boot, n_boot_curve) for a in alphas}}
    pos = np.random.default_rng(SEED).choice(
        np.flatnonzero(test), size=min(n_importance, int(test.sum())), replace=False
    )
    predict = lambda X: models["hgb"].predict_proba(X)[:, 1]
    out["importancia_blocos"] = block_permutation_importance(predict, d.X.iloc[pos], d.y[pos], FEATURE_BLOCKS).to_dict(
        "records"
    )
    singles = {c: [c] for c in d.numeric + d.categorical}
    out["importancia_variaveis"] = block_permutation_importance(predict, d.X.iloc[pos], d.y[pos], singles).to_dict(
        "records"
    )
    return out


def run_x1(raw: pd.DataFrame, d: Data) -> tuple[pd.DataFrame, dict]:
    """Deriva descritiva por trimestre e AUROC de um classificador que separa cada ano de 2019."""
    raw = raw.reset_index(drop=True)
    X, Y = build_features(raw)
    s = build_strata(raw)
    q = quarter_labels(admission_dates(raw))
    df = pd.DataFrame(
        {
            "trimestre": q,
            "obito": Y["morte"],
            "urgencia": s["urgencia"],
            "idade": s["idade"],
            "obstetrica": s["obstetrica"],
            "censurada": s["censurada"],
            "respiratorio": X["cid_capitulo"] == "X",
            "covid": X["cid3"].isin(["B34", "U07"]),
            "raca_sem_info": raw["RACA_COR"].astype(str).str.strip() == "99",
            "cnes": s["cnes"],
        }
    )
    g = df[df["trimestre"] != "NA"].groupby("trimestre")
    table = g[
        ["obito", "urgencia", "idade", "obstetrica", "censurada", "respiratorio", "covid", "raca_sem_info"]
    ].mean()
    table.insert(0, "internacoes", g.size())
    table["hospitais_ativos"] = g["cnes"].nunique()
    years = d.dates.dt.year.to_numpy()
    domain = {}
    for year in sorted(set(years) - {2019}):
        m = np.isin(years, [2019, year])
        Xd, yd = d.X[m], (years[m] == year).astype(int)
        half = np.random.default_rng(SEED).random(len(yd)) < 0.5
        clf = make_hgb(SEED, d.numeric, d.categorical)
        clf.set_params(clf__max_iter=100, clf__early_stopping=False)
        clf.fit(Xd[half], yd[half])
        domain[int(year)] = float(roc_auc_score(yd[~half], clf.predict_proba(Xd[~half])[:, 1]))
    return table, domain


def run_x8(d: Data, alpha: float = ALPHA) -> dict:
    """Martingale de permutabilidade (óbitos e sobreviventes) e controle do O/E por trimestre, nos cenários de
    X4 (pandemia) e X2 (deriva lenta)."""
    out = {}
    for name, setup in (("x4", X4_SETUP), ("x2", X3_SETUP)):
        p_raw = predict_raw(d, in_window(d.dates, *setup["train"]))
        f = fit_window(p_raw, d.y, in_window(d.dates, *setup["base"]), kind=d.calibrator, alpha=alpha)
        end = pd.Timestamp(setup["quarters"][1]) + pd.DateOffset(months=3)
        idx = np.flatnonzero(in_window(d.dates, setup["quarters"][0], end))
        order = idx[np.argsort(d.dates.to_numpy()[idx], kind="stable")]
        p_t, y_t = f.calibrate(p_raw[order]), d.y[order]
        qlab = quarter_labels(d.dates.iloc[order]).to_numpy()
        res = {}
        for cls, label, cal_scores, test_scores in (
            ("obitos", 1, death_scores(f.p_B[f.y_B == 1]), death_scores(p_t[y_t == 1])),
            ("sobreviventes", 0, f.p_B[f.y_B == 0], p_t[y_t == 0]),
        ):
            log_m = mixture_power_martingale_log10(conformal_pvalues(cal_scores, test_scores, seed=SEED))
            ql = qlab[y_t == label]
            alarm = first_alarm(log_m)
            res[cls] = {
                "alarme_no_trimestre": None if alarm is None else str(ql[alarm]),
                "log10_max_por_trimestre": {str(q): float(log_m[ql == q].max()) for q in np.unique(ql)},
            }
        res["oe"] = {
            str(q): oe_control(float(y_t[qlab == q].sum()), float(p_t[qlab == q].sum()), int((qlab == q).sum()))
            for q in np.unique(qlab)
        }
        out[name] = res
    return out


def strategy_summary(results: dict) -> dict:
    """Por experimento e estratégia: |O/E - 1| médio, trimestres com G1 e G2 violadas (após Holm) e fração
    liberada média."""
    out = {}
    for exp in ("x3", "x4"):
        for s, rows in results.get(exp, {}).items():
            out[f"{exp}_{s}"] = {
                "oe_desvio_medio": float(np.mean([abs(r["oe"] - 1) for r in rows])),
                "trimestres_g1_violada": int(sum(holm([r["p_g1"] for r in rows])["reject"])),
                "trimestres_g2_violada": int(sum(holm([r["p_g2"] for r in rows])["reject"])),
                "fracao_liberada_media": float(np.mean([r["fracao_liberada"] for r in rows])),
            }
    return out


def exploratory_hypotheses(results: dict, alpha: float = ALPHA) -> dict:
    """H1, H2, H3 e H5 com os resultados exploratórios disponíveis (2019-2025)."""
    settings = {}
    for exp in ("x3", "x4"):
        for s, rows in results.get(exp, {}).items():
            if s in STRATEGIES:  # variantes de parâmetro ficam fora das hipóteses
                settings[f"{exp}_{s}"] = rows
    if "x2" in results and "x3_a" not in settings:
        settings["x2_a"] = results["x2"]
    out = {}
    if settings:
        out["h1"] = h1(settings, alpha)
        out["h1b"] = h1b(settings, alpha)
        out["h2"] = h2(settings)
    pairs = [
        (ra, rc)
        for exp in ("x3", "x4")
        if exp in results and {"a", "c"} <= set(results[exp])
        for ra, rc in zip(results[exp]["a"], results[exp]["c"])
    ]
    if pairs:
        out["h3"] = h3(pairs)
    if "x5" in results:
        groups = {}
        for r in results["x5"]["cruzado"]:
            if r["variavel"] == "faixa_etaria":
                groups.setdefault(r["nivel"], []).append(r)
        out["h5"] = h5(groups, alpha)
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _to_csv(path: Path, res) -> None:
    """Tabela plana para os experimentos por período (X2 a X5)."""
    if isinstance(res, list):
        pd.DataFrame(res).to_csv(path, index=False)
    elif isinstance(res, dict) and res and all(isinstance(v, list) for v in res.values()):
        pd.DataFrame([{"grupo": k, **r} for k, rows in res.items() for r in rows]).to_csv(path, index=False)


def _save(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.OPEN_DATA)
    ap.add_argument("--experimento", choices=["x1", "x2", "x3", "x4", "x5", "x6", "x8", "todos"], default="todos")
    ap.add_argument("--desfecho", choices=["morte", "perm_prolongada"], default="morte")
    ap.add_argument("--sensibilidade", choices=SENSITIVITIES, default="nenhuma")
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "exploratorios")
    args = ap.parse_args(argv)
    raw = load_open(args.data)
    d = prepare(raw, args.desfecho, args.sensibilidade)
    out_dir = args.out / args.desfecho / args.sensibilidade
    out_dir.mkdir(parents=True, exist_ok=True)
    want = lambda e: args.experimento in (e, "todos")
    runners = {"x2": run_x2, "x3": run_x3, "x4": run_x4, "x5": run_x5, "x6": run_x6, "x8": run_x8}
    if want("x1"):
        table, domain = run_x1(raw, d)
        table.to_csv(out_dir / "x1_deriva_trimestral.csv")
        _save(out_dir / "resultados_x1.json", {"auroc_dominio": domain})
    for exp, run in runners.items():
        if want(exp):
            res = run(d)
            _save(out_dir / f"resultados_{exp}.json", res)
            _to_csv(out_dir / f"{exp}.csv", res)
    results = {
        p.stem.removeprefix("resultados_"): json.loads(p.read_text()) for p in sorted(out_dir.glob("resultados_*.json"))
    }
    summary = {
        "desfecho": args.desfecho,
        "sensibilidade": args.sensibilidade,
        "info": d.info,
        "experimentos_disponiveis": sorted(results),
        "hipoteses_exploratorias": exploratory_hypotheses(results),
        "resumo_estrategias": strategy_summary(results),
    }
    _save(out_dir / "resumo.json", summary)
    print(f"resultados em {out_dir}")


if __name__ == "__main__":
    main()
