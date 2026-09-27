import pandas as pd
import pytest

from triagem_baixo_risco.dataset import (
    SealedDataError,
    admission_dates,
    filter_by_municipio,
    in_window,
    load_open,
    load_sealed,
    months_to_process,
    part_path,
    quarter_labels,
    sealed_path,
    split_sealed,
    to_cod6,
)


def _raw(dates):
    return pd.DataFrame({"DT_INTER": dates, "MORTE": ["0"] * len(dates)})


def test_quarter_labels_use_calendar_quarters():
    d = admission_dates(_raw(["20230115", "20230401", "20231231", "xx"]))
    assert quarter_labels(d).tolist() == ["2023T1", "2023T2", "2023T4", "NA"]


def test_in_window_is_left_closed_and_right_open():
    d = admission_dates(_raw(["20221231", "20230101", "20231231", "20240101"]))
    assert in_window(d, "2023-01-01", "2024-01-01").tolist() == [False, True, True, False]


def test_split_sealed_separates_admissions_from_2026():
    open_df, sealed_df = split_sealed(_raw(["20251231", "20260101", "20240505"]))
    assert open_df["DT_INTER"].tolist() == ["20251231", "20240505"]
    assert sealed_df["DT_INTER"].tolist() == ["20260101"]


def test_load_open_refuses_sealed_rows_and_drops_admissions_before_2019(tmp_path):
    path = tmp_path / "base.parquet"
    _raw(["20181231", "20190101", "20251231"]).to_parquet(path)
    assert load_open(path)["DT_INTER"].tolist() == ["20190101", "20251231"]
    _raw(["20190101", "20260102"]).to_parquet(path)
    with pytest.raises(SealedDataError):
        load_open(path)


def test_sealed_file_sits_next_to_the_open_file(tmp_path):
    path = tmp_path / "sih.parquet"
    assert sealed_path(path).name == "sih_selado.parquet"
    _raw(["20260110"]).to_parquet(sealed_path(path))
    assert load_sealed(path)["DT_INTER"].tolist() == ["20260110"]


def test_to_cod6_strips_ibge_check_digit():
    assert to_cod6(3143302) == 314330  # Montes Claros


def test_filter_by_municipio_keeps_only_rows_in_region():
    df = pd.DataFrame({"MUNIC_MOV": ["314330", "310620"], "x": [1, 2]})
    out = filter_by_municipio(df, codes6={314330}, column="MUNIC_MOV")
    assert out["x"].tolist() == [1]


def test_decode_json_body_handles_gzip_and_plain_bodies():
    import gzip

    from triagem_baixo_risco.dataset import decode_json_body

    assert decode_json_body(gzip.compress(b'[{"id": 3143302}]'), "gzip") == [{"id": 3143302}]
    assert decode_json_body(b"[1]", None) == [1]


def test_months_to_process_skips_months_already_done():
    done = {(2019, 1), (2019, 2)}
    todo = months_to_process([2019], done)
    assert todo[0] == (2019, 3)
    assert len(todo) == 10


def test_part_path_encodes_uf_year_and_month(tmp_path):
    assert part_path(tmp_path, "MG", 2024, 1).name == "MG_202401.parquet"


def test_purge_raw_removes_only_the_requested_month(tmp_path):
    from triagem_baixo_risco.dataset import purge_raw

    d = tmp_path / "downloads" / "ducklake" / "sih"
    d.mkdir(parents=True)
    (d / "RDMG2401.parquet").write_bytes(b"x")
    (d / "RDMG2402.parquet").write_bytes(b"x")
    assert purge_raw(tmp_path, "MG", 2024, 1) == 1
    assert not (d / "RDMG2401.parquet").exists()
    assert (d / "RDMG2402.parquet").exists()


def test_dbc_name_uses_two_digit_year_and_month():
    from triagem_baixo_risco.dataset import dbc_name

    assert dbc_name("MG", 2019, 1) == "RDMG1901.dbc"


def test_normalize_strings_keeps_integers_without_decimal_point():
    import numpy as np
    import pandas as pd

    from triagem_baixo_risco.dataset import normalize_strings

    df = pd.DataFrame(
        {"DIAS_PERM": [6.0, np.nan], "MORTE": [1, 0], "MUNIC_MOV": [" 314330 ", "310620"], "VAL_TOT": [12.5, 3.0]}
    )
    out = normalize_strings(df)
    assert out["DIAS_PERM"].tolist() == ["6", ""]
    assert out["MORTE"].tolist() == ["1", "0"]
    assert out["MUNIC_MOV"].tolist() == ["314330", "310620"]
    assert out["VAL_TOT"].tolist() == ["12.5", "3.0"]


def test_validate_month_rejects_empty_or_incomplete_frames():
    import pandas as pd
    import pytest

    from triagem_baixo_risco.dataset import validate_month

    with pytest.raises(ValueError):
        validate_month(pd.DataFrame())
    with pytest.raises(ValueError):
        validate_month(pd.DataFrame({"MUNIC_MOV": ["314330"]}))


def test_months_to_process_stops_at_the_last_published_month():
    todo = months_to_process([2026], set(), last=(2026, 7))
    assert todo[-1] == (2026, 7) and len(todo) == 7


def test_sha256_file_hashes_content(tmp_path):
    import hashlib

    from triagem_baixo_risco.dataset import sha256_file

    f = tmp_path / "x.dbc"
    f.write_bytes(b"abc")
    assert sha256_file(f) == hashlib.sha256(b"abc").hexdigest()


def test_consolidate_writes_admissions_from_2026_to_the_sealed_file(tmp_path):
    import pandas as pd

    from triagem_baixo_risco.dataset import consolidate, sealed_path

    parts = tmp_path / "parts"
    parts.mkdir()
    pd.DataFrame({"DT_INTER": ["20251220", "20260105"], "N_AIH": ["1", "2"]}).to_parquet(parts / "MG_202601.parquet")
    pd.DataFrame({"DT_INTER": ["20251110"], "N_AIH": ["3"]}).to_parquet(parts / "MG_202512.parquet")
    out = tmp_path / "sih.parquet"
    assert consolidate(parts, "MG", out) == {"open": 2, "sealed": 1}
    assert sorted(pd.read_parquet(out)["N_AIH"]) == ["1", "3"]
    assert pd.read_parquet(sealed_path(out))["N_AIH"].tolist() == ["2"]


def test_process_month_hides_the_deaths_of_sealed_competences(tmp_path, monkeypatch):
    import pandas as pd

    import triagem_baixo_risco.dataset
    from triagem_baixo_risco.dataset import process_month

    month = pd.DataFrame({"MUNIC_MOV": ["314330", "314330", "310620"], "MORTE": ["1", "0", "1"]})
    monkeypatch.setattr(triagem_baixo_risco.dataset, "download_month", lambda *a, **k: month.copy())
    kw = dict(codes6={314330}, parts_dir=tmp_path / "parts", cache_dir=tmp_path / "cache", keep_raw=True)
    assert process_month("MG", 2025, 12, **kw)["deaths_region"] == 1
    info = process_month("MG", 2026, 1, **kw)
    assert info["deaths_region"] is None and info["rows_region"] == 2


def test_validate_month_requires_the_columns_of_the_model_subgroups_and_confirmatory_checks():
    import pandas as pd
    import pytest

    from triagem_baixo_risco.dataset import REQUIRED_COLUMNS, validate_month

    full = pd.DataFrame({c: ["1"] for c in sorted(REQUIRED_COLUMNS)})
    assert validate_month(full) is full
    for col in ("ESPEC", "COBRANCA", "MES_CMPT", "N_AIH"):
        with pytest.raises(ValueError, match=col):
            validate_month(full.drop(columns=col))
