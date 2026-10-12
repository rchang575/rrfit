import numpy as np
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
