from types import SimpleNamespace

import numpy as np
import pytest

from rrfit.dataio import Device, Trace
from rrfit.losserr import (
    compute_loss_errors,
    estimate_sigma_phi,
    is_reference_trace,
    loss_arrays,
    qi_errorbars_from_loss,
)


def _trace(**kwargs):
    defaults = dict(
        power=-40.0, Qi=2e7, Qi_err=2e5, absQc=1e6, phi=0.3, phi_err=1e-3,
        is_excluded=False,
    )
    return Trace(**dict(defaults, **kwargs))


def test_is_reference_trace_default_rules():
    assert is_reference_trace(_trace())
    assert is_reference_trace(_trace(power=-50.0))
    assert not is_reference_trace(_trace(power=-50.1))
    assert not is_reference_trace(_trace(power=None))
    assert not is_reference_trace(_trace(is_excluded=True))
    assert not is_reference_trace(_trace(phi_err=0.0))
    assert not is_reference_trace(_trace(phi=np.nan))

    magnitude = _trace()
    magnitude.fit_method = "magnitude"
    circle = _trace()
    circle.fit_method = "circle"
    assert not is_reference_trace(magnitude)
    assert is_reference_trace(circle)
    assert is_reference_trace(_trace(power=-60.0), min_power=-60.0)


def test_is_reference_trace_tolerates_missing_attributes():
    bare = SimpleNamespace(power=-30.0, phi=0.2)
    assert is_reference_trace(bare)


def test_estimate_sigma_phi_uses_only_reference_traces():
    phis = [0.30, 0.32, 0.27, 0.31]
    traces = [_trace(phi=phi) for phi in phis]
    traces += [
        _trace(phi=2.0, power=-90.0),
        _trace(phi=2.0, is_excluded=True),
        _trace(phi=2.0, phi_err=0.0),
    ]
    device = Device(name="d", traces=traces)

    assert estimate_sigma_phi(device) == pytest.approx(np.std(phis, ddof=1))
    custom = estimate_sigma_phi(device, reference=lambda tr: tr.phi < 0.31)
    assert custom == pytest.approx(np.std([0.30, 0.27], ddof=1))


def test_estimate_sigma_phi_is_floored_at_reference_phi_err():
    agreeing = Device(name="d", traces=[_trace(phi=0.3, phi_err=2e-3), _trace(phi=0.3, phi_err=4e-3)])
    assert estimate_sigma_phi(agreeing) == pytest.approx(3e-3)

    scattered = Device(name="d", traces=[_trace(phi=0.3), _trace(phi=0.4)])
    assert estimate_sigma_phi(scattered) == pytest.approx(np.std([0.3, 0.4], ddof=1))

    no_errors = Device(name="d", traces=[_trace(phi=0.3, phi_err=None), _trace(phi=0.3, phi_err=None)])
    assert estimate_sigma_phi(no_errors) == 0.0


def test_estimate_sigma_phi_needs_two_reference_traces():
    device = Device(name="d", traces=[_trace(), _trace(power=-90.0)])
    with pytest.raises(ValueError, match="at least 2 reference traces"):
        estimate_sigma_phi(device)


def test_compute_loss_errors_formulas_and_no_mutation_of_existing_attrs():
    device = Device(name="d", traces=[_trace(phi=0.3), _trace(phi=0.32, Qi=-5e7)])

    sigma_phi = compute_loss_errors(device)

    assert sigma_phi == device.sigma_phi == pytest.approx(np.std([0.3, 0.32], ddof=1))
    tr = device.traces[0]
    assert (tr.Qi, tr.Qi_err) == (2e7, 2e5)
    assert tr.loss == 1 / 2e7
    assert tr.loss_err_stat == pytest.approx(2e5 / 2e7**2)
    assert tr.loss_err_sys == pytest.approx(np.sin(0.3) / 1e6 * sigma_phi)
    assert tr.loss_err == pytest.approx(np.hypot(tr.loss_err_stat, tr.loss_err_sys))
    # A negative Qi is just a negative loss.
    assert device.traces[1].loss == pytest.approx(-2e-8)


def test_compute_loss_errors_explicit_sigma_phi_and_missing_values():
    device = Device(name="d", traces=[_trace(), _trace(Qi=None)])

    assert compute_loss_errors(device, sigma_phi=0.02) == 0.02
    assert np.isnan(device.traces[1].loss_err)
    with pytest.raises(ValueError, match="finite, positive loss_err"):
        loss_arrays(device.traces)


def test_overcoupled_sanity_numbers():
    # 4p61-like numbers: sigma_phi ~ 0.013 rad gives loss_err_sys ~ 8e-9, so the
    # relative Qi error is ~10% at Qi ~ 1.2e7 and ~30% at Qi ~ 3.4e7.
    abs_qc = 1e6
    phi = np.arcsin(8e-9 / 0.013 * abs_qc)
    device = Device(
        name="d",
        traces=[_trace(Qi=1.2e7, Qi_err=1e5, phi=phi), _trace(Qi=3.4e7, Qi_err=5e5, phi=phi)],
    )

    compute_loss_errors(device, sigma_phi=0.013)

    low, high = device.traces
    assert low.loss_err_sys == pytest.approx(8e-9)
    assert low.loss_err / low.loss == pytest.approx(0.10, abs=0.01)
    assert high.loss_err / high.loss == pytest.approx(0.28, abs=0.03)


def test_loss_arrays_requires_compute_loss_errors():
    with pytest.raises(ValueError, match="compute_loss_errors"):
        loss_arrays([_trace()])


def test_qi_errorbars_from_loss():
    bars = qi_errorbars_from_loss([1e-7, 1e-7], [0.5e-7, 2e-7])

    np.testing.assert_allclose(bars[:, 0], [1e7 - 1 / 1.5e-7, 1 / 0.5e-7 - 1e7])
    np.testing.assert_allclose(bars[:, 1], [1e7 - 1 / 3e-7, 9e7])
