from collections import deque

from lmfit import Parameters

from rrfit.dataio import Device
from rrfit.waterfall import (
    fitIterated,
    fit_waterfall_staged,
    predict_qint_consistent_curve,
)


def _make_params(delta_qp0=0.0, q_tls0=1.0):
    params = Parameters()
    params.add("delta_QP0", value=delta_qp0)
    params.add("Q_TLS0", value=q_tls0)
    params.add("tc", value=2.0)
    params.add("Q_other", value=1e7)
    params.add("beta", value=1.0)
    params.add("beta2", value=1.0)
    params.add("D_0", value=100.0)
    return params


def test_fitIterated_uses_best_fitted_params_for_final_refit(monkeypatch):
    draws = deque([1.0, 2.0])
    call_count = {"value": 0}

    def fake_uniform(_lower, _upper):
        return draws.popleft()

    def fake_fit(device, params, consistent=False, makePlot=True):
        call_count["value"] += 1
        fit_params = params.copy()
        if call_count["value"] <= 2:
            fit_params["delta_QP0"].value = params["delta_QP0"].value + 10.0
            red_chi2 = 2.0 if params["delta_QP0"].value == 1.0 else 1.0
            return fit_params, red_chi2

        # The final refit should start from the best fitted params, not the best seed.
        return fit_params, None, None

    monkeypatch.setattr("rrfit.waterfall.random.uniform", fake_uniform)
    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", fake_fit)

    device = Device(name="device")
    bounds = {"delta_QP0": (0.0, 5.0)}
    init_params = _make_params()

    fitIterated(
        device,
        bounds,
        numIter=2,
        consistent=False,
        makePlot=False,
        init_params=init_params,
    )

    assert device.best_params["delta_QP0"].value == 12.0


def test_predict_qint_consistent_curve_stops_after_bounded_refinement(monkeypatch):
    calls = {"count": 0}

    def fake_consistent(temp, params, freq0, power, qc, qint_init):
        calls["count"] += 1
        if calls["count"] == 1:
            return [1.0, 1.0005]
        return [1.0, 2.0]

    monkeypatch.setattr("rrfit.waterfall.QIntVsTemp_consistent", fake_consistent)

    ys = predict_qint_consistent_curve(
        [0.1, 0.2],
        _make_params(),
        [5.0, 5.0],
        [1.0, 1.0],
        10.0,
        [1.0, 1.0],
        max_refinement_passes=4,
    )

    assert list(ys) == [1.0, 2.0]
    assert calls["count"] == 2


def test_fit_waterfall_staged_runs_fast_search_then_final_consistent_fit(monkeypatch):
    calls = []

    def fake_fit_iterated(
        device,
        boundsDict,
        numIter,
        consistent=False,
        makePlot=True,
        fitQP=True,
        retries=10,
        init_params=None,
    ):
        calls.append(("fitIterated", consistent, numIter))
        best = init_params.copy()
        best["delta_QP0"].value = 3.0 if not consistent else 4.0
        device.best_params = best
        return "search"

    def fake_fit_qint(device, init_params, consistent=False, makePlot=True):
        calls.append(("Fit_QIntVsTemp", consistent, init_params["delta_QP0"].value))
        final_params = init_params.copy()
        final_params["delta_QP0"].value += 1.0
        return final_params, None, None

    monkeypatch.setattr("rrfit.waterfall.fitIterated", fake_fit_iterated)
    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", fake_fit_qint)

    device = Device(name="device")
    bounds = {"delta_QP0": (0.0, 5.0)}

    result = fit_waterfall_staged(
        device,
        bounds,
        coarse_iterations=20,
        refine_iterations=5,
        makePlot=False,
        init_params=_make_params(),
    )

    assert calls == [
        ("fitIterated", False, 20),
        ("fitIterated", True, 5),
        ("Fit_QIntVsTemp", True, 4.0),
    ]
    fit_params, init_fig, fitted_fig = result
    assert fit_params["delta_QP0"].value == 5.0
    assert init_fig is None
    assert fitted_fig is None
    assert device.best_params["delta_QP0"].value == 5.0
    assert device.waterfall_staged_result["coarse_search"] == "search"
    assert device.waterfall_staged_result["consistent_refine"] == "search"


def test_fitIterated_final_refit_handles_makePlot_false(monkeypatch):
    def fake_uniform(_lower, _upper):
        return 1.0

    def fake_fit(device, params, consistent=False, makePlot=True):
        fit_params = params.copy()
        if makePlot:
            return fit_params, None, None
        return fit_params, 0.25

    monkeypatch.setattr("rrfit.waterfall.random.uniform", fake_uniform)
    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", fake_fit)

    device = Device(name="device")
    bounds = {"delta_QP0": (0.0, 5.0)}

    init_dict, final_dict, red_chi2, figures = fitIterated(
        device,
        bounds,
        numIter=1,
        consistent=False,
        makePlot=False,
        init_params=_make_params(),
    )

    assert init_dict["delta_QP0"][0] == 1.0
    assert final_dict["delta_QP0"][0] == 1.0
    assert red_chi2[0] == 0.25
    assert figures == [None, None, None, None, None]
