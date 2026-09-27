from pathlib import Path

import pandas as pd

from triagem_baixo_risco import config


def test_paths_are_absolute_and_under_the_repository_root():
    root = Path(__file__).resolve().parents[1]
    assert config.ROOT == root
    for p in (
        config.DATA_DIR,
        config.RAW_DIR,
        config.INTERIM_DIR,
        config.PROCESSED_DIR,
        config.EXTERNAL_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
        config.OPEN_DATA,
        config.ETL_LOG,
        config.MUNICIPIOS_CSV,
    ):
        assert p.is_absolute() and root in p.parents


def test_cohort_dates_and_seed():
    assert config.COHORT_START == "2019-01-01"
    assert config.SEAL_DATE == pd.Timestamp("2026-01-01")
    assert config.SEED == 42
