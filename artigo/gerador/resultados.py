"""Figuras e tabelas da seção 4 do artigo, lidas dos resultados já salvos (não refaz análise).

Uso, a partir da raiz do repositório: .venv/bin/python artigo/gerador/resultados.py
Grava as figuras em artigo/figuras/ e as tabelas em artigo/texto/tabela-*.md (colunas separadas por " | ").
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FuncFormatter
from scipy.stats import beta

RAIZ = Path(__file__).resolve().parents[2]
EXP = RAIZ / "reports/exploratorios/morte/nenhuma"
CONF = RAIZ / "reports/confirmatorio"
FIG = RAIZ / "artigo/figuras"
TXT = RAIZ / "artigo/texto"
FIG.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

AZUL, VERDE, VERMELHO, CINZA = "#185FA5", "#0F6E56", "#C0392B", "#5F5E5A"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    }
)
CM = 1 / 2.54


def br(x, d=1):
    """Número no formato brasileiro."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def eixo_br(ax, eixo="y", d=1, pct=False):
    f = FuncFormatter(lambda v, _: br(v, d) + ("%" if pct else ""))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(f)


def ic_exato(k, n, conf=0.95):
    t = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(t, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - t, k + 1, n - k)
    return lo, hi


def holm(ps, nivel=0.05):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False] * len(ps)
    for r, i in enumerate(idx):
        if ps[i] <= nivel / (len(ps) - r):
            rej[i] = True
        else:
            break
    return rej


def ler_csv(nome):
    return list(csv.DictReader(open(EXP / nome, encoding="utf8")))


def rotulo_tri(t):
    return f"{t[5]}T{t[2:4]}"


def grava_tabela(nome, linhas):
    (TXT / nome).write_text("\n".join(" | ".join(l) for l in linhas) + "\n", encoding="utf8")


# 4.1: coorte por ano e deriva
x1 = ler_csv("x1_deriva_trimestral.csv")
auroc_ano = json.loads((EXP / "resultados_x1.json").read_text())["auroc_dominio"]
por_ano = defaultdict(lambda: defaultdict(float))
hosp = defaultdict(list)
for r in x1:
    a, n = r["trimestre"][:4], int(r["internacoes"])
    por_ano[a]["n"] += n
    for c in ["obito", "urgencia", "idade", "obstetrica", "respiratorio", "covid", "raca_sem_info"]:
        por_ano[a][c] += n * float(r[c])
    hosp[a].append(int(r["hospitais_ativos"]))
anos = sorted(por_ano)
m = lambda a, c: por_ano[a][c] / por_ano[a]["n"]
grava_tabela(
    "tabela-coorte.md",
    [
        ["Característica"] + anos,
        ["Internações"] + [br(por_ano[a]["n"], 0) for a in anos],
        ["Óbitos (%)"] + [br(100 * m(a, "obito"), 2) for a in anos],
        ["Urgência (%)"] + [br(100 * m(a, "urgencia")) for a in anos],
        ["Idade média (anos)"] + [br(m(a, "idade")) for a in anos],
        ["Obstétricas (%)"] + [br(100 * m(a, "obstetrica")) for a in anos],
        ["Doenças respiratórias (%)"] + [br(100 * m(a, "respiratorio")) for a in anos],
        ["Códigos B34 e U07 (%)"] + [br(100 * m(a, "covid")) for a in anos],
        ["Raça/cor sem informação (%)"] + [br(100 * m(a, "raca_sem_info")) for a in anos],
        ["Hospitais ativos por trimestre"]
        + [f"{min(hosp[a])} a {max(hosp[a])}" if min(hosp[a]) != max(hosp[a]) else str(min(hosp[a])) for a in anos],
        ["AUROC do classificador de ano"] + [".." if a == "2019" else br(auroc_ano[a], 2) for a in anos],
    ],
)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM), gridspec_kw={"width_ratios": [1.6, 1]})
tris = [r["trimestre"] for r in x1]
mort = [100 * float(r["obito"]) for r in x1]
xs = range(len(tris))
i0, i1 = tris.index("2020T2"), tris.index("2021T4")
a1.axvspan(i0 - 0.5, i1 + 0.5, color="#EF9F27", alpha=0.15, lw=0)
a1.text((i0 + i1) / 2, max(mort) * 1.04, "pandemia", ha="center", va="bottom", fontsize=7, color=CINZA)
a1.plot(xs, mort, color=AZUL, marker="o", ms=2.5, lw=1.2)
a1.set_xticks([tris.index(f"{a}T1") for a in anos], anos)
a1.set_ylabel("óbitos por internação (%)")
a1.set_ylim(0, max(mort) * 1.15)
eixo_br(a1, d=0)
a1.set_title("A. Mortalidade por trimestre")
anos_c = [a for a in anos if a != "2019"]
a2.plot(anos_c, [auroc_ano[a] for a in anos_c], color=AZUL, marker="o", ms=3, lw=1.2)
a2.axhline(0.5, color=CINZA, ls="--", lw=0.8)
a2.text(0, 0.515, "0,5: anos indistinguíveis", fontsize=6.5, color=CINZA)
a2.set_ylim(0.45, 0.8)
eixo_br(a2, d=2)
a2.set_ylabel("AUROC")
a2.set_title("B. Quanto cada ano difere de 2019")
fig.tight_layout()
fig.savefig(FIG / "figura-coorte.png")
plt.close(fig)

# 4.2: garantias sob deriva, estratégias (a) e (c)
resumo = json.loads((EXP / "resumo.json").read_text())["hipoteses_exploratorias"]


def serie(arq, g):
    rs = sorted(
        [r for r in ler_csv(arq + ".csv") if r["grupo"] == g and r["alpha_usado"]], key=lambda r: r["trimestre"]
    )
    return rs


fig, eixos = plt.subplots(2, 2, figsize=(16 * CM, 10 * CM), sharey="row")
for col, (arq, titulo) in enumerate(
    [("x3", "2023 a 2025, calibração inicial em 2022"), ("x4", "Pandemia, calibração inicial no fim de 2019")]
):
    for g, cor, mk, nome in [("a", AZUL, "o", "(a) estática"), ("c", VERDE, "s", "(c) trimestral com atraso")]:
        rs = serie(arq, g)
        tr = [rotulo_tri(r["trimestre"]) for r in rs]
        fdl = [100 * float(r["fdl"]) for r in rs]
        razao = [float(r["tol"]) / float(r["r_hat"]) for r in rs]
        v1 = set(resumo["h1"]["cenarios"][f"{arq}_{g}"]["periodos_violados"] or [])
        v2 = holm([float(r["p_g2"]) for r in rs])
        for lin, ys, viol in [(0, fdl, [r["trimestre"] in v1 for r in rs]), (1, razao, v2)]:
            ax = eixos[lin][col]
            ax.plot(range(len(tr)), ys, color=cor, marker=mk, ms=3, lw=1.1, label=nome)
            for i, v in enumerate(viol):
                if v:
                    ax.plot(i, ys[i], marker=mk, ms=7, mfc="none", mec=VERMELHO, mew=1.4)
            ax.set_xticks(range(len(tr)), tr, rotation=90)
    eixos[0][col].axhline(5, color=CINZA, ls="--", lw=0.8)
    eixos[1][col].axhline(1, color=CINZA, ls="--", lw=0.8)
    eixos[0][col].set_title(titulo)
eixos[0][0].set_ylabel("G1: óbitos liberados (%)")
eixos[1][0].set_ylabel("G2: taxa entre as liberadas / r̂")
eixo_br(eixos[0][0], d=0)
eixo_br(eixos[1][0], d=1)
alca = eixos[0][0].get_legend_handles_labels()
# a violação é marcada com o contorno vermelho no formato do ponto da estratégia: círculo em (a), quadrado em (c)
viol_h = tuple(plt.Line2D([], [], marker=m, ms=7, mfc="none", mec=VERMELHO, mew=1.4, ls="none") for m in "os")
fig.legend(
    alca[0] + [viol_h],
    alca[1] + ["violação significativa após Holm"],
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / "figura-garantias.png")
plt.close(fig)

NOMES = {
    "a": "(a) estática",
    "b": "(b) recalibração anual",
    "c": "(c) recalibração trimestral com atraso",
    "d": "(d) retreino anual",
    "e": "(e) conformal adaptativo",
    "f": "(f) janela suave",
    "g": "(g) correção de prevalência",
}
linhas = [
    [
        "Estratégia",
        "G1 violada, 2023 a 2025",
        "G2 violada, 2023 a 2025",
        "|O/E − 1| médio, 2023 a 2025",
        "G1 violada, pandemia",
        "G2 violada, pandemia",
        "|O/E − 1| médio, pandemia",
    ]
]
for g in "abcdefg":
    cel = [NOMES[g]]
    for arq in ["x3", "x4"]:
        rs = serie(arq, g)
        if not rs:
            cel += ["..", "..", ".."]
            continue
        v1 = len(resumo["h1"]["cenarios"][f"{arq}_{g}"]["periodos_violados"] or [])
        v2 = sum(holm([float(r["p_g2"]) for r in rs]))
        oe = sum(abs(float(r["oe"]) - 1) for r in rs) / len(rs)
        cel += [f"{v1} de {len(rs)}", f"{v2} de {len(rs)}", br(oe, 3)]
    linhas.append(cel)
grava_tabela("tabela-estrategias.md", linhas)

# 4.3: equidade (X5, 2023 a 2025 somados) e utilidade (X6, 2025)
x5 = json.loads((EXP / "resultados_x5.json").read_text())
faixas = ["0-17", "18-39", "40-59", "60-79", "80+"]
soma = {}
for regra in ["global", "cruzado"]:
    acc = defaultdict(lambda: [0, 0])
    for r in x5[regra]:
        if r["variavel"] == "faixa_etaria":
            acc[r["nivel"]][0] += r["fdl_k"]
            acc[r["nivel"]][1] += r["fdl_n"]
    soma[regra] = acc
fig, ax = plt.subplots(figsize=(16 * CM, 6.5 * CM))
larg = 0.38
for j, (regra, cor, nome) in enumerate([("global", AZUL, "regra única"), ("cruzado", VERDE, "regra por faixa etária")]):
    ks = [soma[regra][f] for f in faixas]
    vals = [100 * k / n for k, n in ks]
    ics = [ic_exato(k, n) for k, n in ks]
    err = [[v - 100 * lo for v, (lo, _) in zip(vals, ics)], [100 * hi - v for v, (_, hi) in zip(vals, ics)]]
    ax.bar(
        [i + (j - 0.5) * larg for i in range(len(faixas))],
        vals,
        larg,
        color=cor,
        label=nome,
        yerr=err,
        capsize=2,
        error_kw={"lw": 0.8},
    )
ax.axhline(5, color=CINZA, ls="--", lw=0.8)
ax.set_xticks(range(len(faixas)), ["0 a 17", "18 a 39", "40 a 59", "60 a 79", "80 ou mais"])
ax.set_xlabel("faixa etária (anos)")
ax.set_ylabel("óbitos liberados (%)")
eixo_br(ax, d=0)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "figura-equidade.png")
plt.close(fig)

x6 = json.loads((EXP / "resultados_x6.json").read_text())["alfas"]["0.05"]
pc = lambda v: br(100 * v)
regra = x6["regra_sem_modelo"]
grava_tabela(
    "tabela-utilidade.md",
    [
        [
            "Modelo ou regra",
            "Internações liberadas (%)",
            "Adultos não obstétricos liberados (%)",
            "Óbitos liberados (%)",
        ],
        [
            "*Boosting*",
            pc(x6["hgb"]["fracao_liberada"]),
            pc(x6["hgb"]["fracao_liberada_adultos"]),
            pc(x6["hgb"]["fdl"]),
        ],
        [
            "Logística completa",
            pc(x6["logistica"]["fracao_liberada"]),
            pc(x6["logistica"]["fracao_liberada_adultos"]),
            pc(x6["logistica"]["fdl"]),
        ],
        [
            "Logística com 4 variáveis",
            pc(x6["logistica_simples"]["fracao_liberada"]),
            pc(x6["logistica_simples"]["fracao_liberada_adultos"]),
            pc(x6["logistica_simples"]["fdl"]),
        ],
        ["Regra sem modelo", pc(regra["fracao_liberada"]), "...", pc(regra["fdl"])],
        [
            "*Boosting* com a fração de óbitos liberados da regra",
            pc(regra["hgb_mesma_fdl_fracao"]),
            "...",
            pc(regra["hgb_mesma_fdl_fdl"]),
        ],
    ],
)

# 4.4: rodada confirmatória
x7 = {v: json.loads((CONF / f"{v}.json").read_text()) for v in ["x7a", "x7b"]}


def met(v, chave):
    return x7[v]["alfas"]["0.05"]["metricas"][chave]


def ic_txt(par, d, fator=1):
    return f"{br(fator * par[0], d)} a {br(fator * par[1], d)}"


linhas = [["Medida", "X7a", "X7b"]]
linhas.append(["Internações no teste"] + [br(x7[v]["n_teste"], 0) for v in x7])
linhas.append(["Óbitos no teste"] + [br(x7[v]["obitos_teste"], 0) for v in x7])
linhas.append(["Mortalidade no teste (%)"] + [br(100 * met(v, "prevalencia"), 2) for v in x7])
linhas.append(["Mortalidade no teste / mortalidade na parte B"] + [br(met(v, "razao_prevalencia"), 2) for v in x7])
linhas.append(["AUROC (IC 95%)"] + [f"{br(met(v, 'auroc'), 3)} ({ic_txt(met(v, 'auroc_ic'), 3)})" for v in x7])
linhas.append(["O/E (IC 95%)"] + [f"{br(met(v, 'oe'), 2)} ({ic_txt(met(v, 'oe_ic'), 2)})" for v in x7])
linhas.append(["Internações liberadas (%)"] + [br(100 * met(v, "fracao_liberada")) for v in x7])
linhas.append(
    ["Adultos não obstétricos liberados (%)"]
    + [br(100 * x7[v]["alfas"]["0.05"]["utilidade"]["hgb"]["fracao_liberada_adultos"]) for v in x7]
)
linhas.append(
    ["G1: óbitos liberados (%, IC 95%)"]
    + [
        f"{br(100 * met(v, 'fdl'), 2)} ({ic_txt(x7[v]['alfas']['0.05']['h1']['cenarios'][v]['agregado']['ic'], 2, 100)})"
        for v in x7
    ]
)
linhas.append(["G2: taxa de óbito entre as liberadas (%)"] + [br(100 * met(v, "tol"), 2) for v in x7])
linhas.append(["Limite r̂ (%)"] + [br(100 * met(v, "r_hat"), 2) for v in x7])
for chave, nome in [
    ("regra", "Diferença contra a regra sem modelo, com a mesma fração de óbitos liberados (p.p., IC 95%)"),
    ("logistica_simples", "Diferença contra a logística com 4 variáveis (p.p., IC 95%)"),
]:
    linhas.append(
        [nome]
        + [
            f"{br(100 * x7[v]['alfas']['0.05']['utilidade']['diferencas'][chave]['diferenca'])} ({ic_txt(x7[v]['alfas']['0.05']['utilidade']['diferencas'][chave]['ic'], 1, 100)})"
            for v in x7
        ]
    )
grava_tabela("tabela-confirmatoria.md", linhas)

dec = lambda v, h: "não refutada" if x7[v]["alfas"]["0.05"][h]["refutada"] is False else "refutada"
grava_tabela(
    "tabela-decisoes.md",
    [
        ["Hipótese", "X7a", "X7b"],
        ["H1", dec("x7a", "h1"), dec("x7b", "h1")],
        ["H1b", dec("x7a", "h1b"), dec("x7b", "h1b")],
        ["H2", dec("x7a", "h2"), dec("x7b", "h2")],
        ["H4", dec("x7a", "h4"), dec("x7b", "h4")],
        ["H5", "descritiva", "não avaliada"],
    ],
)
print("figuras:", sorted(p.name for p in FIG.glob("*.png")))
print("tabelas:", sorted(p.name for p in TXT.glob("tabela-*.md")))
