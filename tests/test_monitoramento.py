import numpy as np

from triagem_baixo_risco.monitoramento import conformal_pvalues, first_alarm, mixture_power_martingale_log10, oe_control


def test_conformal_pvalues_are_uniform_under_exchangeability():
    rng = np.random.default_rng(0)
    p = conformal_pvalues(rng.random(5000), rng.random(20000))
    assert abs(p.mean() - 0.5) < 0.01 and p.min() > 0 and p.max() <= 1


def test_martingale_stays_low_without_shift_and_alarms_after_a_shift():
    rng = np.random.default_rng(1)
    cal = rng.random(5000)
    quiet = mixture_power_martingale_log10(conformal_pvalues(cal, rng.random(20000), seed=2))
    assert first_alarm(quiet) is None
    shifted = np.concatenate([rng.random(2000), rng.random(3000) ** 0.5])  # escores maiores depois de 2000
    log_m = mixture_power_martingale_log10(conformal_pvalues(cal, shifted, seed=3))
    alarm = first_alarm(log_m)
    assert alarm is not None and alarm >= 2000


def test_oe_control_flags_only_large_deviations():
    assert oe_control(100, 100, 2000)["alarme"] is False
    assert oe_control(150, 100, 2000)["alarme"] is True
