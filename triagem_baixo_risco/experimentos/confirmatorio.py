"""Rodada confirmatória X7: única parte do código que lê as internações lacradas.

Uso:
    python -m triagem_baixo_risco.experimentos.confirmatorio --variante a

Cada variante roda uma vez. Reexecutar exige --desvio "motivo": a rodada grava um arquivo novo e acrescenta o
motivo a desvios.md na pasta de saída.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

import triagem_baixo_risco
from triagem_baixo_risco import config
from triagem_baixo_risco.avaliacao import evaluate_period, h1, h1b, h4, h5, residual_g1
from triagem_baixo_risco.dataset import (
    SEAL_DATE,
    SealedDataError,
    admission_dates,
    in_window,
    load_open,
    load_sealed,
    sealed_path,
    sha256_file,
)
from triagem_baixo_risco.experimentos.exploratorios import (
    SENSITIVITIES,
    _json_default,
    fit_baselines,
    prepare,
    utility_comparisons,
)
from triagem_baixo_risco.modeling.predict import apply_rule
from triagem_baixo_risco.modeling.train import Fitted, fit_window

VARIANTS = {
    "a": {
        "train": ("2019-01-01", "2025-01-01"),
        "cal": ("2025-01-01", "2026-01-01"),
        "test": ("2026-01-01", "2026-04-01"),
        "cmpt_ajuste": "2026-07",
        "cmpt_teste": "2026-07",
    },
    "b": {
        "train": ("2019-01-01", "2024-10-01"),
        "cal": ("2024-10-01", "2025-10-01"),
        "test": ("2026-01-01", "2026-04-01"),
        "cmpt_ajuste": "2026-07",
        "cmpt_teste": "2026-07",
    },
    "c": {
        "train": ("2019-01-01", "2025-01-01"),
        "cal": ("2025-01-01", "2026-01-01"),
        "test": ("2026-04-01", "2026-07-01"),
        "cmpt_ajuste": "2026-07",
        "cmpt_teste": "2026-10",
    },
}
H5_SCOPE = {"a": "descritiva", "b": "nao_avaliada", "c": "confirmatoria"}
FIRST_CMPT = "2019-01"
ALPHAS = (0.05, 0.10)
MIN_DEATHS_H5 = 100
FOLLOWUP_LIMIT = 0.90


def check_followup_completeness(raw: pd.DataFrame, start, end, limit: float = FOLLOWUP_LIMIT) -> dict:
    """Razão de seguimento por mês de internação no teste contra o mesmo mês do ano anterior; meses sem
    referência ficam fora. Recusa se algum mês ficar abaixo de limit. Devolve as razões por mês."""
    dates = admission_dates(raw)
    end_month = pd.Timestamp(end).to_period("M")
    if pd.Timestamp(end).is_month_start:
        end_month -= 1
    months = pd.period_range(pd.Timestamp(start).to_period("M"), end_month, freq="M")
    ratios = {}
    for m in months:
        this = int(((dates.dt.year == m.year) & (dates.dt.month == m.month)).sum())
        prev = int(((dates.dt.year == m.year - 1) & (dates.dt.month == m.month)).sum())
        if prev:
            ratios[f"{m.year}-{m.month:02d}"] = this / prev
    bad = {k: v for k, v in ratios.items() if v < limit}
    if bad:
        detail = ", ".join(f"{k}: {v:.2f}" for k, v in bad.items())
        raise ValueError(f"seguimento incompleto (razão abaixo de {limit:.2f}): {detail}")
    return ratios


def h2_conditional(metrics: dict, high: float = 1.2, low: float = 1.0) -> dict:
    """Previsão condicional de H2 em 2026: G2 vale se a razão de prevalência <= low; falha se >= high."""
    ratio, violated = metrics["razao_prevalencia"], metrics["p_g2"] < 0.05
    if ratio <= low:
        return {"previsao": "G2 vale", "refutada": bool(violated), "razao": ratio}
    if ratio >= high:
        return {"previsao": "G2 falha", "refutada": bool(not violated), "razao": ratio}
    return {"previsao": "sem previsão", "refutada": None, "razao": ratio}


def check_integrity(raw: pd.DataFrame, sealed: pd.DataFrame) -> dict:
    """Antes de qualquer ajuste: toda linha lacrada com DT_INTER a partir de SEAL_DATE e, na junção dos dois arquivos
    (raw), N_AIH único e IDENT = 1 (se a coluna existir)."""
    ds = admission_dates(sealed)
    bad = int((ds.isna() | (ds < SEAL_DATE)).sum())
    if bad:
        raise SealedDataError(f"{bad} linhas do arquivo lacrado sem DT_INTER a partir de {SEAL_DATE.date()}")
    dup = int(raw["N_AIH"].astype(str).str.strip().duplicated().sum())
    if dup:
        raise ValueError(f"{dup} N_AIH repetido(s) entre os arquivos aberto e lacrado")
    has_ident = "IDENT" in raw.columns
    if has_ident and (other := int((raw["IDENT"].astype(str).str.strip() != "1").sum())):
        raise ValueError(f"{other} linhas com IDENT diferente de 1")
    return {
        "lacradas_a_partir_de": str(SEAL_DATE.date()),
        "linhas_lacradas": int(len(sealed)),
        "n_aih_unicos": int(len(raw)),
        "ident": "todas 1" if has_ident else "coluna ausente",
    }


def _yyyymm(cmpt: str) -> int:
    return int(cmpt.replace("-", ""))


def competence_filter(raw: pd.DataFrame, v: dict) -> tuple[np.ndarray, dict]:
    """Corte de competência da variante: linhas do teste até cmpt_teste e as demais até cmpt_ajuste. Antes do corte,
    confere em todas as linhas (aberto e lacrado) que o ETL entregou cada mês de FIRST_CMPT ao maior corte. Devolve a
    máscara e as contagens das linhas mantidas."""
    cmpt = (raw["ANO_CMPT"].astype(int) * 100 + raw["MES_CMPT"].astype(int)).to_numpy()
    present = set(np.unique(cmpt).tolist())
    months = pd.period_range(FIRST_CMPT, max(v["cmpt_ajuste"], v["cmpt_teste"]), freq="M")
    missing = [str(m) for m in months if m.year * 100 + m.month not in present]
    if missing:
        raise ValueError(f"competências ausentes entre {FIRST_CMPT} e o corte: {', '.join(missing)}")
    in_test = in_window(admission_dates(raw), *v["test"])
    keep = (in_test & (cmpt <= _yyyymm(v["cmpt_teste"]))) | (~in_test & (cmpt <= _yyyymm(v["cmpt_ajuste"])))
    counts = pd.Series(cmpt[keep]).value_counts().sort_index()
    label = lambda c: f"{c // 100}-{c % 100:02d}"
    return keep, {
        "primeira": label(counts.index[0]),
        "ultima": label(counts.index[-1]),
        "linhas_por_competencia": {label(c): int(n) for c, n in counts.items()},
        "linhas_fora_do_corte": int((~keep).sum()),
    }


def _rule_record(hgb: Fitted, cross: Fitted) -> dict:
    r = hgb.rule
    return {
        "hgb": {
            "q1": r.q1,
            "r_hat": r.r_hat,
            "n1": r.n1,
            "n_liberadas_cal": r.n_released_cal,
            "obitos_liberados_cal": r.deaths_released_cal,
        },
        "cruzado": {"q1": cross.rule.q1, "n1": cross.rule.n1},
    }


def rule_mismatches(ref: dict, new: dict) -> list[str]:
    """Campos em que a regra refeita difere da registrada: q1, r_hat e n1 da regra global; q1 e n1 por faixa."""
    new = json.loads(json.dumps(new, default=_json_default))  # mesmas chaves (texto) e tipos do arquivo
    fields = [(f"hgb.{k}", ref["hgb"][k], new["hgb"][k]) for k in ("q1", "r_hat", "n1")]
    fields += [
        (f"cruzado.{k}[{b}]", ref["cruzado"][k].get(b), new["cruzado"][k].get(b))
        for k in ("q1", "n1")
        for b in sorted(set(ref["cruzado"][k]) | set(new["cruzado"][k]))
    ]
    same = lambda a, b: a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return [f"{name}: {a} != {b}" for name, a, b in fields if not same(a, b)]


def code_version(root: Path) -> dict:
    """Commit do código (git rev-parse HEAD) e se a árvore de trabalho tinha mudanças (git status --porcelain não vazio,
    arquivo não rastreado incluído). Os dois ficam None sem git, sem commit ou se root não for a raiz do repositório que
    o git acha: uma cópia sem .git dentro de outro repositório daria o commit do outro."""
    git = lambda *args: subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout
    try:
        top, commit = git("rev-parse", "--show-toplevel", "HEAD").splitlines()
        if Path(top).resolve() == Path(root).resolve():
            return {"commit": commit, "arvore_suja": git("status", "--porcelain") != ""}
    except (OSError, subprocess.CalledProcessError):
        pass  # sem git, fora de um repositório ou sem commit
    return {"commit": None, "arvore_suja": None}


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pacote": triagem_baixo_risco.__file__,
        **code_version(config.ROOT),
    }


def run_confirmatory(
    open_path,
    variant: str,
    out_dir,
    target: str = "morte",
    deviation: str | None = None,
    n_boot: int = 1000,
    sensitivity: str = "nenhuma",
) -> dict:
    """Uma variante de X7 (ou uma sensibilidade de X7a), gravada sem sobrescrever nenhuma saída anterior."""
    if sensitivity != "nenhuma" and variant != "a":
        raise ValueError("sensibilidades só em X7a")
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"sensibilidade desconhecida: {sensitivity}")
    out_dir, v = Path(out_dir), VARIANTS[variant]
    base = f"x7{variant}" + ("" if sensitivity == "nenhuma" else f"_{sensitivity}")
    label = f"X7{variant}" + ("" if sensitivity == "nenhuma" else f" {sensitivity}")
    if (out_dir / f"{base}.json").exists() and not deviation:
        raise RuntimeError(f"{label} já foi executada; para reexecutar, declare o desvio com --desvio")
    if not sealed_path(open_path).exists():
        raise FileNotFoundError(f"arquivo lacrado não encontrado ({sealed_path(open_path)}); rode antes o ETL de 2026")
    ref_file = None
    if variant == "c":
        refs = [p for p in out_dir.glob("x7a*.json") if p.name == "x7a.json" or p.name.startswith("x7a_desvio_")]
        if not refs:
            raise RuntimeError("X7c usa a regra congelada de X7a: rode X7a antes")
        ref_file = max(refs, key=lambda p: p.name)  # o desvio mais recente, se houver
    started = time.localtime()
    environment = _environment()  # antes de ler os dados e de gravar qualquer saída
    sealed = load_sealed(open_path)
    raw = pd.concat([load_open(open_path), sealed], ignore_index=True)  # sem guardar outra cópia do aberto
    integrity = check_integrity(raw, sealed)
    keep, competences = competence_filter(raw, v)
    check_followup_completeness(raw, *v["test"])
    d = prepare(raw[keep], target, sensitivity)
    train, cal, test = (in_window(d.dates, *v[k]) for k in ("train", "cal", "test"))
    _, preds = fit_baselines(d, train)
    faixa = d.strata["faixa_etaria"].astype(str).to_numpy()
    fits = {
        str(a): (
            fit_window(preds["hgb"], d.y, cal, kind=d.calibrator, alpha=a),
            fit_window(preds["hgb"], d.y, cal, kind=d.calibrator, alpha=a, groups=faixa),
        )
        for a in ALPHAS
    }
    rules = {a: _rule_record(hgb, cross) for a, (hgb, cross) in fits.items()}
    result = {
        "variante": variant,
        "sensibilidade": sensitivity,
        "janelas": v,
        "desfecho": target,
        "ambiente": environment,
        "executado_em": time.strftime("%Y-%m-%dT%H:%M:%S", started),
        "desvio": deviation,
        "sha256_dados": {"aberto": sha256_file(open_path), "lacrado": sha256_file(sealed_path(open_path))},
        "integridade": integrity,
        "competencias": competences,
        "info_preparo": d.info,
        "escopo_h5": H5_SCOPE[variant],
        "n_teste": int(test.sum()),
        "obitos_teste": int(d.y[test].sum()),
        "alfas": {},
    }
    if ref_file is not None:
        ref = json.loads(ref_file.read_text())
        bad = [f"alfa {a}, {m}" for a in rules for m in rule_mismatches(ref["alfas"][a]["regra"], rules[a])]
        if bad:
            raise RuntimeError(f"a regra refeita em X7c difere da registrada em {ref_file.name}: " + "; ".join(bad))
        result["regra_de_referencia"] = ref_file.name
    for a, (hgb, cross) in fits.items():
        alpha = float(a)
        util = utility_comparisons(d, preds, cal, test, alpha, n_boot, n_boot_curve=100 if n_boot >= 100 else 0)
        m = dict(util["metricas_hgb"])
        rel_hgb = apply_rule(hgb.rule, hgb.calibrate(preds["hgb"][test]))
        m.update(
            residual_g1(
                d.y, rel_hgb, d.codes, d.dates, pd.Timestamp(v["test"][0]), np.flatnonzero(test), alpha, n1=hgb.rule.n1
            )
        )
        p_t = cross.calibrate(preds["hgb"][test])
        rel = apply_rule(cross.rule, p_t, faixa[test])
        faixa_b, groups = faixa[cross.index_B], {}
        for band in np.unique(faixa[test]):
            mk, y_b = faixa[test] == band, cross.y_B[faixa_b == band]
            prev_b = float(y_b.mean()) if len(y_b) else float("nan")  # prevalência da faixa na parte B
            groups[band] = [
                evaluate_period(
                    d.y[test][mk], p_t[mk], rel[mk], alpha, cross.rule.r_hat, prev_b, n1=cross.rule.n1.get(band, 0)
                )
            ]
        diffs = {k: tuple(util["diferencas"][k]["ic"]) for k in ("regra", "logistica_simples")}
        result["alfas"][a] = {
            "regra": rules[a],
            "metricas": m,
            "utilidade": util,
            "h1": h1({base: [m]}, alpha),
            "h1b": h1b({base: [m]}, alpha),
            "h2": h2_conditional(m),
            "h4": h4(util["hgb"]["fracao_liberada"], util["hgb"]["fracao_liberada_adultos"], diffs),
            "h5": h5(groups, alpha, min_deaths=MIN_DEATHS_H5),
            "cruzado_por_faixa": {k: g[0] for k, g in groups.items()},
        }
    exists = (out_dir / f"{base}.json").exists()
    result["arquivo"] = f"{base}_desvio_{time.strftime('%Y%m%dT%H%M%S', started)}.json" if exists else f"{base}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / result["arquivo"], "x") as fh:  # "x": nunca sobrescreve uma saída
        fh.write(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
    if deviation:
        with open(out_dir / "desvios.md", "a") as fh:
            fh.write(f"- {result['executado_em']}, {label}: {deviation} (arquivo {result['arquivo']})\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.OPEN_DATA)
    ap.add_argument("--variante", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--sensibilidade", choices=SENSITIVITIES, default="nenhuma", help="só com --variante a")
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "confirmatorio")
    ap.add_argument("--desvio", default=None, help="motivo, se for reexecução")
    args = ap.parse_args(argv)
    res = run_confirmatory(args.data, args.variante, args.out, deviation=args.desvio, sensitivity=args.sensibilidade)
    for alpha, r in res["alfas"].items():
        h5_txt = {"a": f" | H5 (descritiva)={r['h5']['refutada']}", "c": f" | H5 refutada={r['h5']['refutada']}"}
        print(
            f"alfa {alpha}: H1 refutada={r['h1']['refutada']} | H2 {r['h2']} | H4 refutada={r['h4']['refutada']}"
            f"{h5_txt.get(args.variante, '')}"
        )
    print(f"resultados em {args.out / res['arquivo']}")


if __name__ == "__main__":
    main()
