"""Figuras a partir dos resultados salvos pelos experimentos exploratórios.

Uso:
    python -m triagem_baixo_risco.plots --resultados reports/exploratorios/morte/nenhuma

Figuras em reports/figures/<desfecho>/, a não ser que --saida seja informado.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triagem_baixo_risco import config
from triagem_baixo_risco.modeling.predict import exact_ci

LABELS = {
    "a": "(a) estática",
    "b": "(b) anual",
    "c": "(c) trimestral com atraso",
    "d": "(d) retreino anual",
    "e": "(e) conformal adaptativo",
    "f": "(f) janela suave",
    "g": "(g) correção de prevalência",
}
# Paleta de ordem fixa, escolhida para que cores vizinhas se distingam também com daltonismo; o marcador identifica a
# estratégia sem depender da cor.
STYLES = {
    "a": dict(color="#2a78d6", marker="o", ls="-"),
    "b": dict(color="#eb6834", marker="s", ls="-"),
    "c": dict(color="#1baf7a", marker="^", ls="-"),
    "d": dict(color="#eda100", marker="D", ls="-"),
    "e": dict(color="#e87ba4", marker="v", ls="-"),
    "f": dict(color="#008300", marker="P", ls="-"),
    "g": dict(color="#4a3aa7", marker="X", ls="-"),
}
VARIANT_STYLES = (dict(marker="s", ls="--"), dict(marker="D", ls=":"), dict(marker="^", ls="-."))  # cor da estratégia
PARAMETER_NAMES = {"gamma": "γ = {}", "meia_vida": "meia-vida de {} trimestres"}
PANEL_TITLES = {
    "morte": ("G1: fração dos óbitos liberados", "G2: óbito entre as liberadas menos o limite", "O/E"),
    "perm_prolongada": (
        "G1: fração das permanências prolongadas liberadas",
        "G2: permanência prolongada entre as liberadas menos o limite",
        "O/E",
    ),
}
OE_MAX = 3.0
INK = "#333333"


def panel_titles(desfecho: str) -> tuple[str, str, str]:
    """Títulos dos painéis G1, G2 e O/E para o desfecho."""
    if desfecho not in PANEL_TITLES:
        raise ValueError(f"desfecho desconhecido: {desfecho}")
    return PANEL_TITLES[desfecho]


def _comma(x: float, places: int = 2) -> str:
    return f"{x:.{places}f}".replace(".", ",")


def variant_keys(rows_by_strategy: dict) -> list[str]:
    """Variantes de parâmetro, como e(gamma=0.01) ou f(meia_vida=8): as chaves que não são estratégias de LABELS."""
    return [k for k in rows_by_strategy if k not in LABELS]


def _split_variant(key: str) -> tuple[str, str, str]:
    base, _, inner = key.partition("(")
    name, _, value = inner.rstrip(")").partition("=")
    return base, name, value


def _variant_label(key: str) -> str:
    base, name, value = _split_variant(key)
    return f"({base}) " + PARAMETER_NAMES.get(name, name + " = {}").format(value.replace(".", ","))


def _variant_order(key: str) -> tuple[float, str]:
    try:
        return float(_split_variant(key)[2]), key
    except ValueError:
        return math.inf, key


def _clip_oe(ax, series, pos: dict) -> None:
    """Eixo do O/E em [0, OE_MAX]; cada ponto acima vira um marcador na borda de cima, anotado com valor, estratégia e
    trimestre, por exemplo "7,22 (g, 2024T2)"."""
    ax.set_ylim(0, OE_MAX)
    center = (len(pos) - 1) / 2
    stacked = {}
    for name, _, rows, style, _ in series:
        for r in rows:
            if not r["oe"] > OE_MAX:
                continue
            x = pos[r["trimestre"]]
            ax.plot(
                [x], [OE_MAX], marker="^", ms=7, color=style["color"], mec="white", mew=0.6, clip_on=False, zorder=4
            )
            left = x < center
            level = stacked.get(left, 0)
            stacked[left] = level + 1
            ax.annotate(
                f"{_comma(r['oe'])} ({name}, {r['trimestre']})",
                xy=(x, OE_MAX),
                xytext=(-7 if left else 7, 3 + 10 * level),
                textcoords="offset points",
                ha="right" if left else "left",
                va="bottom",
                fontsize=7,
                color=INK,
                annotation_clip=False,
            )


def _garantias_figure(series, alpha: float, title: str, path: Path, desfecho: str, legend_keys: list[str]) -> None:
    """Três painéis por trimestre; series = [(chave, rótulo, linhas, estilo, destaque)], na ordem de desenho."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    quarters = sorted({r["trimestre"] for _, _, rows, _, _ in series for r in rows})
    pos = {q: i for i, q in enumerate(quarters)}
    handles = {}
    for name, label, rows, style, strong in series:
        x = [pos[r["trimestre"]] for r in rows]
        kw = dict(
            color=style["color"],
            marker=style["marker"],
            ls=style["ls"],
            lw=2.0 if strong else 1.2,
            ms=5.0 if strong else 4.0,
            mec="white",
            mew=0.6,
            zorder=3 if strong else 2,
        )
        handles[name] = axes[0].plot(x, [r["fdl"] for r in rows], label=label, **kw)[0]
        axes[1].plot(x, [r["tol"] - r["r_hat"] for r in rows], **kw)
        axes[2].plot(x, [r["oe"] for r in rows], **kw)
    for ax, ref, name in zip(axes, (alpha, 0.0, 1.0), panel_titles(desfecho)):
        ax.axhline(ref, color="k", ls="--", lw=1, zorder=1)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(quarters)), quarters)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_xlim(-0.5, len(quarters) - 0.5)
    _clip_oe(axes[2], series, pos)
    fig.legend(
        [handles[k] for k in legend_keys],
        [handles[k].get_label() for k in legend_keys],
        loc="lower center",
        ncol=len(legend_keys),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_garantias(rows_by_strategy: dict, alpha: float, title: str, path: Path, desfecho: str = "morte") -> None:
    """G1, G2 (excesso sobre o limite) e O/E por trimestre, uma linha por estratégia de (a) a (g). A estática (a) vai
    por último e por cima, com traço mais grosso; as variantes de parâmetro ficam em fig_parametros."""
    keys = [s for s in LABELS if s in rows_by_strategy]
    order = [s for s in keys if s != "a"] + [s for s in keys if s == "a"]
    series = [(s, LABELS[s], rows_by_strategy[s], STYLES[s], s == "a") for s in order]
    _garantias_figure(series, alpha, title, path, desfecho, keys)


def fig_parametros(rows_by_strategy: dict, alpha: float, title: str, path: Path, desfecho: str = "morte") -> None:
    """As estratégias com variantes de parâmetro, (e) e (f), com as variantes: a cor é a da estratégia, e o traço e o
    marcador distinguem cada variante."""
    series = []
    for base in sorted({_split_variant(k)[0] for k in variant_keys(rows_by_strategy)}):
        style = STYLES.get(base, dict(color=INK, marker="o", ls="-"))
        if base in rows_by_strategy:
            series.append((base, f"{LABELS.get(base, base)} (padrão)", rows_by_strategy[base], style, False))
        variants = sorted(
            (k for k in variant_keys(rows_by_strategy) if _split_variant(k)[0] == base), key=_variant_order
        )
        for k, extra in zip(variants, itertools.cycle(VARIANT_STYLES)):
            series.append((k, _variant_label(k), rows_by_strategy[k], {**style, **extra}, False))
    _garantias_figure(series, alpha, title, path, desfecho, [name for name, *_ in series])


def fig_equidade(x5: dict, alpha: float, path: Path) -> None:
    """FDL agregada de 2023 a 2025 por faixa etária, Mondrian por classe contra Mondrian cruzado, com IC exato de 95%
    (Clopper-Pearson) sobre os k e n somados."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bands = sorted({r["nivel"] for r in x5["global"] if r["variavel"] == "faixa_etaria"})
    width = 0.38
    for i, name in enumerate(("global", "cruzado")):
        vals, below, above = [], [], []
        for b in bands:
            rows = [r for r in x5[name] if r["variavel"] == "faixa_etaria" and r["nivel"] == b]
            k, n = sum(r["fdl_k"] for r in rows), sum(r["fdl_n"] for r in rows)
            v = k / n if n else np.nan
            lo, hi = exact_ci(k, n)
            vals.append(v)
            below.append(v - lo)
            above.append(hi - v)
        ax.bar(
            np.arange(len(bands)) + (i - 0.5) * width,
            vals,
            width,
            yerr=[below, above],
            capsize=3,
            error_kw=dict(lw=1, ecolor=INK),
            label="Mondrian por classe" if name == "global" else "Mondrian cruzado",
        )
    ax.axhline(alpha, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(bands)), bands)
    ax.set_ylabel("fração dos óbitos liberados")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibracao(curve: dict, path: Path) -> None:
    """Curva de calibração suavizada com zoom em 0-0,2 e histogramas por desfecho."""
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(curve["p"], curve["observado_suavizado"], label="curva suavizada")
    if "banda_inferior" in curve:
        ax.fill_between(curve["p"], curve["banda_inferior"], curve["banda_superior"], alpha=0.25, label="IC 95%")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.2)
    ax.set_ylabel("frequência observada")
    ax.legend(fontsize=8)
    edges = np.asarray(curve["bordas_histograma"])
    centers = (edges[:-1] + edges[1:]) / 2
    axh.bar(centers, curve["histograma_sobreviventes"], width=np.diff(edges), alpha=0.6, label="sobreviventes")
    axh.bar(centers, curve["histograma_obitos"], width=np.diff(edges), alpha=0.8, label="óbitos")
    axh.set_yscale("log")
    axh.set_xlabel("probabilidade prevista")
    axh.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fdl_label(v) -> str:
    return "FDL indefinida" if v is None or not np.isfinite(v) else f"FDL {_comma(100 * v, 1)}%"


def fig_utilidade(res: dict, path: Path) -> None:
    """Fração de todas as internações do teste liberada pelo boosting e pelas linhas de base, com a FDL de cada regra
    no teste ao lado da barra."""
    names = ["hgb", "logistica", "logistica_simples"]
    labels = [
        "boosting",
        "logística completa",
        "logística de 4 variáveis",
        "regra sem modelo",
        "boosting (mesma FDL da regra)",
    ]
    rule = res["regra_sem_modelo"]
    todas = [res[n]["fracao_liberada"] for n in names] + [rule["fracao_liberada"], rule["hgb_mesma_fdl_fracao"]]
    fdl = [res[n]["fdl"] for n in names] + [rule["fdl"], rule["hgb_mesma_fdl_fdl"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels[::-1], todas[::-1])
    ax.bar_label(bars, labels=[_fdl_label(v) for v in fdl[::-1]], padding=4, fontsize=8, color=INK)
    ax.set_xlim(0, max(todas) * 1.25)
    ax.set_xlabel("fração das internações liberadas")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(results_dir: Path, alpha: float = 0.05, out_dir: Path | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"pasta de resultados não encontrada: {results_dir}")
    if not any(results_dir.glob("resultados_*.json")):
        raise FileNotFoundError(
            f"nenhum resultados_*.json em {results_dir}; gere-os com "
            "python -m triagem_baixo_risco.experimentos.exploratorios"
        )
    summary = results_dir / "resumo.json"
    desfecho = json.loads(summary.read_text()).get("desfecho", "morte") if summary.exists() else "morte"
    out = config.FIGURES_DIR / desfecho if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    load = lambda name: json.loads((results_dir / f"resultados_{name}.json").read_text())
    for exp, title in (("x3", "Deriva lenta, 2023-2025"), ("x4", "Pandemia, 2020-2021")):
        if (results_dir / f"resultados_{exp}.json").exists():
            rows = load(exp)
            fig_garantias(rows, alpha, title, out / f"garantias_{exp}.png", desfecho)
            made.append(out / f"garantias_{exp}.png")
            if variant_keys(rows):
                fig_parametros(
                    rows,
                    alpha,
                    f"{title}: variantes de parâmetro de (e) e (f)",
                    out / f"garantias_{exp}_parametros.png",
                    desfecho,
                )
                made.append(out / f"garantias_{exp}_parametros.png")
    if (results_dir / "resultados_x5.json").exists():
        fig_equidade(load("x5"), alpha, out / "equidade_x5.png")
        made.append(out / "equidade_x5.png")
    if (results_dir / "resultados_x6.json").exists():
        res = load("x6")["alfas"][str(alpha)]
        fig_calibracao(res["curva_calibracao"], out / "calibracao_x6.png")
        fig_utilidade(res, out / "utilidade_x6.png")
        made += [out / "calibracao_x6.png", out / "utilidade_x6.png"]
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", type=Path, default=config.REPORTS_DIR / "exploratorios" / "morte" / "nenhuma")
    ap.add_argument("--alfa", type=float, default=0.05)
    ap.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args(argv)
    for p in make_all(args.resultados, args.alfa, args.saida):
        print(p)


if __name__ == "__main__":
    main()
