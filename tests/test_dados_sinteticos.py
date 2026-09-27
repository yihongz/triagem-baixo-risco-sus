from tests.dados_sinteticos import synthetic_raw


def test_synthetic_rows_carry_fields_for_linkage_and_strata():
    raw = synthetic_raw(n=300, seed=0)
    assert {"DT_SAIDA", "NASC", "CEP", "COBRANCA"} <= set(raw.columns)
    assert set(raw["COBRANCA"]) <= {"41", "61", "12", "31", "27"}
    assert (raw["DT_SAIDA"] >= raw["DT_INTER"]).all()


def test_logit_shift_raises_mortality_only_in_the_shifted_year():
    base = synthetic_raw(n=30000, seed=1)
    shifted = synthetic_raw(n=30000, seed=1, logit_shift={2021: 1.0})
    rate = lambda r, y: (r.loc[r["ANO_CMPT"] == y, "MORTE"] == "1").mean()
    assert rate(shifted, "2021") > rate(base, "2021") + 0.05
    assert rate(shifted, "2019") == rate(base, "2019")
