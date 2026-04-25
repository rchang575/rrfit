import numpy as np

from rrfit.fitfns import asymmetric_lorentzian, rr_s21_hanger
from rrfit.hangerfit import fit_s21
from rrfit.magfit import fit_magnitude


def test_fit_magnitude_recovers_frequency_and_quality_factors():
    f = np.linspace(4.995e9, 5.005e9, 1201)
    true_fr = 5.0e9
    true_phi = 0.12
    true_ql = true_fr / 2.0e6
    true_absqc = 3.3e4
    height = true_ql / true_absqc
    s21_mag = asymmetric_lorentzian(
        f,
        ofs=0.0,
        height=height,
        phi=true_phi,
        fr=true_fr,
        fwhm=true_fr / true_ql,
    )

    result = fit_magnitude(s21_mag, f)
    params = result.params.valuesdict()

    assert np.isclose(params["fr"], true_fr, rtol=1e-4)
    assert params["Ql"] > 0
    assert params["absQc"] > 0
    assert params["Qi"] > 0


def test_fit_s21_returns_expected_parameters_for_synthetic_trace():
    f = np.linspace(4.995e9, 5.005e9, 1201)
    true_params = {
        "fr": 5.0e9,
        "Ql": 2.5e4,
        "absQc": 4.0e4,
        "phi": 0.05,
    }
    s21 = rr_s21_hanger(f, **true_params)

    fit_params = fit_s21(s21.copy(), f)

    assert set(
        [
            "fr",
            "fr_err",
            "Qi",
            "Qi_err",
            "absQc",
            "absQc_err",
            "Ql",
            "Ql_err",
            "phi",
            "phi_err",
            "background_amp",
            "background_phase",
            "chisqr",
            "redchi",
        ]
    ) <= set(fit_params)
    assert np.isfinite(fit_params["fr"])
    assert fit_params["Qi"] > 0
    assert fit_params["Ql"] > 0
    assert fit_params["absQc"] > 0
    assert np.isclose(fit_params["fr"], true_params["fr"], rtol=5e-4)
