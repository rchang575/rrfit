import numpy as np
import pytest
from lmfit import Parameters

from rrfit.QTLS0fit import _store_qtls0_fit_summary
from rrfit.dataio import Device


class DummyResult:
    def __init__(self, params):
        self.params = params


def test_store_qtls0_fit_summary_sets_summary_compatible_attrs():
    params = Parameters()
    params.add("qtls0", value=1.2e6)
    params.add("nc", value=10.0)
    params.add("beta", value=0.7)
    params.add("Qother", value=5e6)
    params["qtls0"].stderr = 1.0e5
    params["nc"].stderr = None
    result = DummyResult(params)
    device = Device(name="device")

    _store_qtls0_fit_summary(device, result)

    assert device.qtls0_fit_result is result
    assert device.qtls0_best_values["qtls0"] == 1.2e6
    assert device.qtls0_best_values["Qother"] == 5e6
    assert device.qtls0_param_errors["qtls0"] == 1.0e5
    assert np.isnan(device.qtls0_param_errors["nc"])


def _make_qtls0_device(qi_err_scale):
    import matplotlib

    matplotlib.use("Agg")
    from rrfit.dataio import Trace
    from rrfit.fitfns import Qivsnbar, dBmtoW, nbarvsPin

    fr, temp, qc, attenuation = 5e9, 0.015, 2e5, 40.0
    powers = np.linspace(-140.0, -60.0, 9)
    traces = []
    for i, power in enumerate(powers):
        # Solve Qi self-consistently with nbar(Ql) so the synthetic data match the model.
        qi = 1e5
        for _ in range(200):
            ql = 1 / (1 / qi + 1 / qc)
            nbar = nbarvsPin(dBmtoW(power - attenuation), fr, ql, qc)
            qi = Qivsnbar(nbar, 2e5, 10.0, 0.8, 3e6, fr, temp)
        qi *= 1 + 0.02 * (-1) ** i
        traces.append(
            Trace(
                power=power, temperature=temp, fr=fr, Qi=qi,
                Qi_err=qi * qi_err_scale[i], Ql=1 / (1 / qi + 1 / qc),
                absQc=qc, phi=0.0, is_excluded=False,
            )
        )
    device = Device(name="device", traces=traces)
    device.attenuation = attenuation
    return device


def test_fit_qtls0_weights_by_qi_err_by_default():
    from rrfit.QTLS0fit import fit_qtls0

    # Errors differ wildly between points, so weighting must change the fit.
    scale = np.array([1e-3, 0.5, 1e-3, 0.5, 1e-3, 0.5, 1e-3, 0.5, 1e-3])
    weighted_default = _make_qtls0_device(scale)
    weighted_explicit = _make_qtls0_device(scale)
    unweighted = _make_qtls0_device(scale)

    fit_qtls0(weighted_default)
    fit_qtls0(weighted_explicit, weighted=True)
    fit_qtls0(unweighted, weighted=False)

    np.testing.assert_allclose(
        weighted_default.qtls0_fit_result.weights,
        1 / np.array([tr.Qi_err for tr in sorted(weighted_default.traces, key=lambda t: t.power)]),
    )
    assert weighted_default.qtls0 == weighted_explicit.qtls0
    assert unweighted.qtls0_fit_result.weights is None
    assert weighted_default.qtls0 != unweighted.qtls0


def test_fit_qtls0_weighted_rejects_missing_errors():
    import pytest
    from rrfit.QTLS0fit import fit_qtls0

    device = _make_qtls0_device(np.full(9, 0.01))
    device.traces[3].Qi_err = 0.0

    with pytest.raises(ValueError, match="positive Qi_err"):
        fit_qtls0(device)


def _fit_quietly(device):
    import contextlib
    import io

    from rrfit.QTLS0fit import fit_qtls0

    with contextlib.redirect_stdout(io.StringIO()):
        fit_qtls0(device)
    return device


def test_fit_qtls0_uses_device_line_attenuation():
    from rrfit.fitfns import dBmtoW, nbarvsPin

    device = _make_qtls0_device(np.full(9, 0.01))
    attenuation = device.attenuation
    del device.attenuation
    device.line_attenuation = attenuation

    _fit_quietly(device)

    traces = sorted(device.traces, key=lambda tr: tr.power)
    expected_nbar = nbarvsPin(
        np.array([dBmtoW(tr.power - attenuation) for tr in traces]),
        np.mean([tr.fr for tr in traces]),
        np.array([tr.Ql for tr in traces]),
        np.mean([tr.absQc for tr in traces]),
    )
    np.testing.assert_allclose(device.qtls0_fit_result.userkws["x"], expected_nbar)


def test_fit_qtls0_line_attenuation_matches_legacy_attenuation_and_rescales_nc():
    legacy = _fit_quietly(_make_qtls0_device(np.full(9, 0.01)))

    modern = _make_qtls0_device(np.full(9, 0.01))
    modern.line_attenuation = modern.attenuation
    del modern.attenuation
    _fit_quietly(modern)

    unattenuated = _make_qtls0_device(np.full(9, 0.01))
    del unattenuated.attenuation
    _fit_quietly(unattenuated)

    np.testing.assert_allclose(
        modern.qtls0_fit_result.userkws["x"], legacy.qtls0_fit_result.userkws["x"]
    )
    assert modern.qtls0 == pytest.approx(legacy.qtls0, rel=1e-9)
    # Ignoring 40 dB of attenuation inflates nbar, and so nc, by 1e4; qtls0 is unaffected.
    np.testing.assert_allclose(
        unattenuated.qtls0_fit_result.userkws["x"],
        1e4 * modern.qtls0_fit_result.userkws["x"],
    )
    assert unattenuated.qtls0 == pytest.approx(modern.qtls0, rel=1e-3)
    assert unattenuated.qtls0_best_values["nc"] == pytest.approx(
        1e4 * modern.qtls0_best_values["nc"], rel=1e-3
    )


def test_fit_qtls0_loss_space_requires_loss_errors():
    from synthetic import overcoupled_qtls0_device
    from rrfit.QTLS0fit import fit_qtls0

    with pytest.raises(ValueError, match="compute_loss_errors"):
        fit_qtls0(overcoupled_qtls0_device(), loss_space=True)


def test_fit_qtls0_loss_space_on_overcoupled_device():
    import contextlib
    import copy
    import io

    from synthetic import QTLS0_TRUE, overcoupled_qtls0_device
    from rrfit.QTLS0fit import fit_qtls0
    from rrfit.losserr import compute_loss_errors

    device = overcoupled_qtls0_device(seed=0)
    compute_loss_errors(device)
    qi_before = [(tr.Qi, tr.Qi_err) for tr in device.traces]
    default = copy.deepcopy(device)

    with contextlib.redirect_stdout(io.StringIO()):
        fit_qtls0(default)
        fit_qtls0(device, loss_space=True)

    traces = sorted(device.traces, key=lambda tr: tr.power)
    result = device.qtls0_fit_result
    np.testing.assert_allclose(result.data, [1 / tr.Qi for tr in traces])
    np.testing.assert_allclose(result.weights, [1 / tr.loss_err for tr in traces])
    assert [(tr.Qi, tr.Qi_err) for tr in device.traces] == qi_before
    # Qi_err ignores the phi systematic, so the default fit is badly overdispersed;
    # the loss-space errors describe the scatter and recover the true qtls0.
    assert default.qtls0_fit_result.redchi > 100
    assert result.redchi < 3
    assert abs(device.qtls0 - QTLS0_TRUE["qtls0"]) < 3 * device.qtls0_err
    assert set(device.qtls0_best_values) == set(default.qtls0_best_values)
