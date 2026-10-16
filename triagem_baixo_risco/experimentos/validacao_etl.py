"""Validacao da base: concordancia da COMPLEX com o procedimento modal.

Confere a afirmativa do artigo e de docs/dados.md: entre as internacoes de 2019 a
2025 da coorte, a COMPLEX da AIH coincide com a mais frequente dentro do mesmo
PROCEDIMENTO REALIZADO em 99,6% dos casos (com o PROCEDIMENTO SOLICITADO, 97,6%).
Grava reports/validacao_etl.json com os numeros, a base usada e o SHA-256 dela.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from triagem_baixo_risco.dataset import admission_dates


def modal_concordancia(sub: pd.DataFrame, col) -> float:
    modal = sub.groupby(col)["COMPLEX"].transform(lambda s: s.value_counts().idxmax())
    return float((sub["COMPLEX"] == modal).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="data/processed/sih_norte_minas.parquet")
    ap.add_argument("--saida", default="reports/validacao_etl.json")
    ap.add_argument("--executado-em", default=None, help="ISO local; padrao e agora")
    args = ap.parse_args()

    df = pd.read_parquet(args.base)
    d = admission_dates(df)
    sub = df[((d >= pd.Timestamp("2019-01-01")) & (d < pd.Timestamp("2026-01-01"))).to_numpy()].copy()
    out = {
        "descricao": "Concordancia da COMPLEX com o procedimento modal (base 2019-2025 da coorte).",
        "executado_em": args.executado_em or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base": args.base,
        "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "linhas_base": int(len(df)),
        "linhas_2019_2025": int(len(sub)),
        "concordancia_complex_procedimento_realizado": round(modal_concordancia(sub, sub["PROC_REA"]), 6),
        "concordancia_complex_procedimento_solicitado": round(modal_concordancia(sub, sub["PROC_SOLIC"]), 6),
    }
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
