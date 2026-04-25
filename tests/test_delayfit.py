import numpy as np

from rrfit.delayfit import fit_cable_delay
from rrfit.fitfns import cable_delay_linear


def test_fit_cable_delay_recovers_tau_without_exclusion():
    f = np.linspace(4.99e9, 5.01e9, 400)
    tau = 4.2e-9
    theta = -0.35
    phase = cable_delay_linear(f, tau=tau, theta=theta)

    fit_tau = fit_cable_delay(phase, f)

    assert np.isclose(fit_tau, tau, rtol=1e-6)


def test_fit_cable_delay_recovers_tau_with_exclusion_window():
    f = np.linspace(4.99e9, 5.01e9, 400)
    tau = 2.8e-9
    theta = 0.6
    phase = cable_delay_linear(f, tau=tau, theta=theta)

    fit_tau = fit_cable_delay(phase, f, exclude=(120, 280))

    assert np.isclose(fit_tau, tau, rtol=1e-6)
