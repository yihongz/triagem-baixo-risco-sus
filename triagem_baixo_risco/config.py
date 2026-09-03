"""Caminhos padrão e constantes de coorte usados em todo o pacote."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

OPEN_DATA = PROCESSED_DIR / "sih_norte_minas.parquet"
ETL_LOG = INTERIM_DIR / "etl_log.jsonl"
MUNICIPIOS_CSV = EXTERNAL_DIR / "municipios_norte_minas.csv"

COHORT_START = "2019-01-01"
SEAL_DATE = pd.Timestamp("2026-01-01")
SEED = 42
