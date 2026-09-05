"""ETL do SIH-SUS no Norte de Minas: baixa os meses, separa a coorte lacrada a partir de 2026 e localiza os
municípios da mesorregião (IBGE 3102).

Uso:
    python -m triagem_baixo_risco.dataset --uf MG --years 2019 2026 --by mov --last-month 2026-07

Se for interrompido, retoma de onde parou: os meses já processados são pulados.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from triagem_baixo_risco import config

SEAL_DATE = config.SEAL_DATE
COHORT_START = pd.Timestamp(config.COHORT_START)
SEALED_SUFFIX = "_selado"


class SealedDataError(RuntimeError):
    """Internações lacradas apareceram onde só dados exploratórios são permitidos."""


def admission_dates(raw: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(raw["DT_INTER"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")


def quarter_labels(dates: pd.Series) -> pd.Series:
    """Rótulo do trimestre civil, por exemplo "2023T1"; "NA" para datas inválidas."""
    lab = dates.dt.year.astype("Int64").astype(str) + "T" + dates.dt.quarter.astype("Int64").astype(str)
    return lab.where(dates.notna(), "NA")


def in_window(dates: pd.Series, start, end) -> np.ndarray:
    """Máscara de datas em [start, end)."""
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy()


def split_sealed(raw: pd.DataFrame, seal_date=SEAL_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    sealed = (admission_dates(raw) >= pd.Timestamp(seal_date)).to_numpy()
    return raw[~sealed].copy(), raw[sealed].copy()


def assert_unsealed(raw: pd.DataFrame, seal_date=SEAL_DATE) -> None:
    n = int((admission_dates(raw) >= pd.Timestamp(seal_date)).sum())
    if n:
        raise SealedDataError(
            f"{n} internações lacradas (a partir de {pd.Timestamp(seal_date).date()}) nos dados exploratórios"
        )


def sealed_path(open_path) -> Path:
    p = Path(open_path)
    return p.with_name(p.stem + SEALED_SUFFIX + p.suffix)


def load_open(path, start=COHORT_START) -> pd.DataFrame:
    """Dados exploratórios: internações a partir de start, recusando qualquer internação lacrada. Linhas sem data
    válida ficam, para serem contadas e descartadas na preparação dos dados."""
    raw = pd.read_parquet(path)
    assert_unsealed(raw)
    dates = admission_dates(raw)
    return raw[~(dates < pd.Timestamp(start)).to_numpy()].reset_index(drop=True)


def load_sealed(open_path) -> pd.DataFrame:
    """Só para a rodada confirmatória."""
    return pd.read_parquet(sealed_path(open_path))


IBGE_MESORREGIAO_NORTE_DE_MINAS = 3102
MUNICIPIOS_CACHE = config.MUNICIPIOS_CSV


def to_cod6(cod7) -> int:
    """SIH usa o código IBGE de 6 dígitos (sem o dígito verificador)."""
    return int(str(int(cod7))[:6])


def decode_json_body(body: bytes, content_encoding: str | None):
    """A API do IBGE pode responder gzip mesmo sem pedido; urllib não descomprime sozinho."""
    if content_encoding and "gzip" in content_encoding.lower():
        body = gzip.decompress(body)
    elif body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return json.loads(body.decode("utf-8"))


def fetch_norte_minas_municipios(cache_path: Path = MUNICIPIOS_CACHE) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    url = (
        f"https://servicodados.ibge.gov.br/api/v1/localidades/mesorregioes/{IBGE_MESORREGIAO_NORTE_DE_MINAS}/municipios"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "triagem-baixo-risco/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = decode_json_body(resp.read(), resp.headers.get("Content-Encoding"))
    df = pd.DataFrame(
        {
            "cod7": [m["id"] for m in data],
            "nome": [m["nome"] for m in data],
            "microrregiao": [m["microrregiao"]["nome"] for m in data],
        }
    )
    df["cod6"] = df["cod7"].map(to_cod6)
    df = df.sort_values("nome").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def filter_by_municipio(df: pd.DataFrame, codes6, column: str = "MUNIC_MOV") -> pd.DataFrame:
    codes = {str(int(c)) for c in codes6}
    return df[df[column].astype(str).str.strip().isin(codes)].copy()


FTP_BASE = "ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados/"
REQUIRED_COLUMNS = {
    "MUNIC_MOV",
    "MUNIC_RES",
    "MORTE",
    "DIAS_PERM",
    "IDADE",
    "COD_IDADE",
    "DIAG_PRINC",
    "DT_INTER",
    "CAR_INT",
    "SEXO",
    "PROC_SOLIC",
    "CNES",
    "ANO_CMPT",
    "ESPEC",
    "COBRANCA",
    "MES_CMPT",
    "N_AIH",
}

DEFAULT_PARTS = config.INTERIM_DIR
DEFAULT_CACHE = config.RAW_DIR
DEFAULT_OUT = config.OPEN_DATA
DEFAULT_LOG = config.ETL_LOG


def months_to_process(years, done, last: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Meses ainda não processados; com last=(ano, mês), para no último mês publicado."""
    return [(y, m) for y in years for m in range(1, 13) if (y, m) not in done and (last is None or (y, m) <= last)]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def part_path(parts_dir, uf: str, year: int, month: int) -> Path:
    return Path(parts_dir) / f"{uf}_{year}{month:02d}.parquet"


def done_months(parts_dir, uf: str) -> set[tuple[int, int]]:
    done = set()
    for f in Path(parts_dir).glob(f"{uf}_*.parquet"):
        m = re.fullmatch(rf"{uf}_(\d{{4}})(\d{{2}})\.parquet", f.name)
        if m:
            done.add((int(m.group(1)), int(m.group(2))))
    return done


def dbc_name(uf: str, year: int, month: int) -> str:
    return f"RD{uf}{year % 100:02d}{month:02d}.dbc"


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Todas as colunas como texto, no formato do DATASUS: inteiros sem '.0', nulos como ''."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_float_dtype(col):
            nn = col.dropna()
            if len(nn) and np.all(np.equal(np.mod(nn, 1), 0)):
                out[c] = col.astype("Int64").astype(str).mask(col.isna(), "")
            else:
                out[c] = col.astype(str).mask(col.isna(), "")
        elif pd.api.types.is_integer_dtype(col):
            out[c] = col.astype(str)
        else:
            out[c] = col.astype(str).str.strip().mask(col.isna(), "")
    return out


def validate_month(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("resposta vazia")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colunas ausentes: {sorted(missing)}")
    return df


def download_dbc(uf: str, year: int, month: int, cache_dir: Path) -> Path:
    name = dbc_name(uf, year, month)
    dest = Path(cache_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(FTP_BASE + name, timeout=180) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    if dest.stat().st_size < 1000:
        dest.unlink(missing_ok=True)
        raise ValueError(f"download de {name} veio vazio")
    return dest


def read_dbc(path: Path, keep_raw: bool = False) -> pd.DataFrame:
    import pyreaddbc
    from dbfread import DBF

    dbf = Path(path).with_suffix(".dbf")
    pyreaddbc.dbc2dbf(str(path), str(dbf))
    try:
        df = pd.DataFrame(iter(DBF(str(dbf), encoding="latin-1", char_decode_errors="replace")))
    finally:
        dbf.unlink(missing_ok=True)
        if not keep_raw:
            Path(path).unlink(missing_ok=True)
    return normalize_strings(df)


def download_month_pysus(uf: str, year: int, month: int, cache_dir: Path) -> pd.DataFrame:
    os.environ["PYSUS_CACHEPATH"] = str(Path(cache_dir).resolve())
    import pysus
    import pysus.ftp

    pysus.set_cache(str(Path(cache_dir).resolve()))
    pysus.disable_progress_bars()
    return pysus.ftp.sih(uf, year, month, group="RD", as_dataframe=True)


def download_month(uf: str, year: int, month: int, cache_dir: Path, keep_raw: bool = False) -> pd.DataFrame:
    """FTP oficial + pyreaddbc; se falhar, PySUS. Sempre valida linhas e colunas obrigatórias."""
    errors = []
    try:
        path = download_dbc(uf, year, month, cache_dir)
        digest = sha256_file(path)
        df = validate_month(read_dbc(path, keep_raw=keep_raw))
        df.attrs.update(source="ftp", sha256=digest)
        return df
    except Exception as exc:
        errors.append(f"ftp: {exc!r}")
    try:
        df = validate_month(download_month_pysus(uf, year, month, cache_dir))
        df.attrs.update(source="pysus", sha256=None)
        return df
    except Exception as exc:
        errors.append(f"pysus: {exc!r}")
    raise RuntimeError(" | ".join(errors))


def _cache_roots(cache_dir: Path) -> list[Path]:
    """PySUS 2.11 grava em ~/pysus/downloads mesmo depois de set_cache; olhar os dois lugares."""
    roots = [Path(cache_dir)]
    try:
        import pysus

        roots.append(Path(pysus.CACHEPATH))
    except Exception:
        pass
    roots.append(Path.home() / "pysus")
    seen, out = set(), []
    for r in roots:
        r = r.resolve()
        if r.exists() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def purge_raw(cache_dir: Path, uf: str, year: int, month: int, roots=None) -> int:
    """Apaga dados brutos cujo nome contém RD{UF}{aa}{mm} nas raízes dadas (padrão: só cache_dir)."""
    tag = f"RD{uf}{year % 100:02d}{month:02d}".lower()
    removed = 0
    for root in roots if roots is not None else [Path(cache_dir)]:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda p: len(str(p)), reverse=True):
            if path.suffix.lower() in {".parquet", ".dbc", ".dbf", ".csv"} or path.is_dir():
                if tag in path.name.lower() and path.exists():
                    shutil.rmtree(path) if path.is_dir() else path.unlink()
                    removed += 1
    return removed


def process_month(
    uf, year, month, codes6, by="mov", parts_dir=DEFAULT_PARTS, cache_dir=DEFAULT_CACHE, keep_raw=False
) -> dict:
    t0 = time.time()
    df = download_month(uf, year, month, cache_dir, keep_raw=keep_raw)
    column = "MUNIC_MOV" if by == "mov" else "MUNIC_RES"
    sub = filter_by_municipio(df, codes6, column).astype(str)
    out = part_path(parts_dir, uf, year, month)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out, index=False)
    removed = 0 if keep_raw else purge_raw(cache_dir, uf, year, month, roots=_cache_roots(cache_dir))
    return {
        "uf": uf,
        "year": year,
        "month": month,
        "by": by,
        "source": df.attrs.get("source"),
        "sha256": df.attrs.get("sha256"),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows_uf": int(len(df)),
        "rows_region": int(len(sub)),
        # óbitos das competências lacradas ficam ocultos até a rodada confirmatória
        "deaths_region": None if year >= SEAL_DATE.year else int((sub["MORTE"].str.strip() == "1").sum()),
        "raw_files_removed": removed,
        "seconds": round(time.time() - t0, 1),
    }


def consolidate(parts_dir, uf: str, out_path: Path, seal_date=SEAL_DATE) -> dict:
    """Junta as partes mensais. Internações com data a partir de seal_date vão para o arquivo lacrado."""
    files = sorted(Path(parts_dir).glob(f"{uf}_*.parquet"))
    if not files:
        return {"open": 0, "sealed": 0}
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    open_df, sealed_df = split_sealed(df, seal_date)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    open_df.to_parquet(out_path, index=False)
    if len(sealed_df):
        sealed_df.to_parquet(sealed_path(out_path), index=False)
    return {"open": int(len(open_df)), "sealed": int(len(sealed_df))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uf", default="MG")
    ap.add_argument("--years", nargs=2, type=int, metavar=("INICIO", "FIM"), default=(2019, 2025))
    ap.add_argument("--months", nargs="*", type=int, default=None, help="meses específicos (padrão: todos)")
    ap.add_argument("--last-month", default=None, help="último mês publicado, AAAA-MM (evita tentar meses futuros)")
    ap.add_argument("--by", choices=["mov", "res"], default="mov", help="mov = município do hospital; res = residência")
    ap.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no-consolidate", action="store_true")
    args = ap.parse_args(argv)

    codes6 = set(fetch_norte_minas_municipios()["cod6"])
    years = range(args.years[0], args.years[1] + 1)
    last = tuple(int(x) for x in args.last_month.split("-")) if args.last_month else None
    todo = months_to_process(years, done_months(args.parts_dir, args.uf), last)
    if args.months:
        todo = [(y, m) for (y, m) in todo if m in set(args.months)]
    print(f"{len(todo)} meses a processar; {len(codes6)} municípios na região", flush=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    for i, (y, m) in enumerate(todo, 1):
        for attempt in range(1, args.retries + 1):
            try:
                info = process_month(args.uf, y, m, codes6, args.by, args.parts_dir, args.cache_dir, args.keep_raw)
                break
            except Exception as exc:  # rede do DATASUS oscila; tentar de novo
                info = {"uf": args.uf, "year": y, "month": m, "error": repr(exc), "attempt": attempt}
                print(f"[{y}-{m:02d}] erro (tentativa {attempt}): {exc}", flush=True)
                time.sleep(10 * attempt)
        with open(args.log, "a") as fh:
            fh.write(json.dumps(info, ensure_ascii=False) + "\n")
        if "error" not in info:
            deaths = (
                "óbitos ocultos até a rodada confirmatória"
                if info["deaths_region"] is None
                else f"{info['deaths_region']} óbitos"
            )
            print(
                f"[{i}/{len(todo)}] {y}-{m:02d}: {info['rows_region']} internações na região "
                f"({deaths}) de {info['rows_uf']} em {args.uf}, {info['seconds']}s",
                flush=True,
            )
    if not args.no_consolidate:
        n = consolidate(args.parts_dir, args.uf, args.out)
        print(
            f"consolidado: {n['open']} linhas em {args.out}; {n['sealed']} lacradas em {sealed_path(args.out)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
