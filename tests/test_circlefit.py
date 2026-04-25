import numpy as np

from rrfit.circlefit import fit_background, fit_circle
from rrfit.fitfns import rr_s21_hanger


def test_fit_circle_recovers_radius_and_center_from_exact_circle():
    radius = 2.5
    center = 1.2 - 0.7j
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    s21 = center + radius * np.exp(1j * theta)

    fit_radius, fit_center = fit_circle(s21)

    assert np.isclose(fit_radius, radius, rtol=1e-6)
    assert np.isclose(fit_center.real, center.real, rtol=1e-6)
    assert np.isclose(fit_center.imag, center.imag, rtol=1e-6)


def test_fit_background_returns_near_unity_for_unity_background_model():
    f = np.linspace(4.995e9, 5.005e9, 1001)
    s21 = rr_s21_hanger(f, fr=5.0e9, Ql=2.4e4, absQc=4.0e4, phi=0.08)

    off_resonant_point = fit_background(s21, f)

    assert np.isclose(np.abs(off_resonant_point), 1.0, atol=5e-2)
    assert np.isclose(np.angle(off_resonant_point), 0.0, atol=5e-2)
