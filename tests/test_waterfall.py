from collections import deque

from lmfit import Parameters
import numpy as np
import pytest

from rrfit.dataio import Device
from rrfit.fitfns import dBmtoW
from rrfit.waterfall import (
    _get_waterfall_arrays,
    _solve_consistent_qint,
    _store_waterfall_fit_summary,
    fitIterated,
    fit_waterfall_staged,
    predict_qint_consistent_curve,
    QIntVsTemp_consistent,
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


def test_store_waterfall_fit_summary_sets_summary_compatible_attrs():
    params = _make_params(delta_qp0=2.5, q_tls0=3.0)
    params["delta_QP0"].stderr = 0.25
    params["Q_TLS0"].stderr = None
    device = Device(name="device")

    _store_waterfall_fit_summary(device, params)

    assert device.waterfall_best_values["delta_QP0"] == 2.5
    assert device.waterfall_best_values["Q_TLS0"] == 3.0
    assert device.waterfall_param_errors["delta_QP0"] == 0.25
    assert np.isnan(device.waterfall_param_errors["Q_TLS0"])


def test_get_waterfall_arrays_accepts_legacy_attenuation_attr():
    trace = type(
        "Trace",
        (),
        {
            "is_excluded": False,
            "power": -80.0,
            "temperature": 0.1,
            "fr": 5e9,
            "Qi": 1e5,
            "Qi_err": 1e3,
            "absQc": 2e4,
            "Ql": 1.67e4,
        },
    )()
    device = Device(name="device", traces=[trace])
    device.attenuation = 70.0

    arrays = _get_waterfall_arrays(device)

    assert arrays["power_watts"][0] == dBmtoW(-150.0)


def test_fitIterated_uses_best_fitted_params_for_final_refit(monkeypatch):
    draws = deque([1.0, 2.0])
    call_count = {"value": 0}

    def fake_uniform(_lower, _upper):
        return draws.popleft()

    def fake_fit(device, params, consistent=False, makePlot=True, inner_solver="local"):
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

    def fake_consistent(
        temp,
        params,
        freq0,
        power,
        qc,
        qint_init,
        inner_solver="local",
    ):
        calls["count"] += 1
        assert inner_solver == "local"
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
        inner_solver="local",
    ):
        calls.append(("fitIterated", consistent, numIter, inner_solver))
        best = init_params.copy()
        best["delta_QP0"].value = 3.0 if not consistent else 4.0
        device.best_params = best
        return "search"

    def fake_fit_qint(
        device,
        init_params,
        consistent=False,
        makePlot=True,
        inner_solver="local",
    ):
        calls.append(
            ("Fit_QIntVsTemp", consistent, init_params["delta_QP0"].value, inner_solver)
        )
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
        ("fitIterated", False, 20, "local"),
        ("fitIterated", True, 5, "local"),
        ("Fit_QIntVsTemp", True, 4.0, "local"),
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

    def fake_fit(device, params, consistent=False, makePlot=True, inner_solver="local"):
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


def test_solve_consistent_qint_prefers_fast_root_solver(monkeypatch):
    calls = []

    class RootResult:
        converged = True
        root = 42.0

    def fake_root_scalar(fn, method, x0, x1, maxiter):
        calls.append((method, x0, x1, maxiter, fn(42.0)))
        return RootResult()

    monkeypatch.setattr("rrfit.waterfall.optimize.root_scalar", fake_root_scalar)

    qint = _solve_consistent_qint(
        _make_params(delta_qp0=1e-6, q_tls0=1e6),
        temp=0.1,
        freq0=5e9,
        power=1e-12,
        Qc=2e4,
        qint_init=1e5,
    )

    assert qint == 42.0
    assert calls[0][0] == "secant"


def test_solve_consistent_qint_falls_back_to_bounded_minimizer(monkeypatch):
    class RootFailure:
        converged = False
        root = np.nan

    class MinResult:
        success = True
        x = 24.0

    monkeypatch.setattr(
        "rrfit.waterfall.optimize.root_scalar",
        lambda *args, **kwargs: RootFailure(),
    )
    monkeypatch.setattr(
        "rrfit.waterfall.optimize.minimize_scalar",
        lambda *args, **kwargs: MinResult(),
    )

    qint = _solve_consistent_qint(
        _make_params(delta_qp0=1e-6, q_tls0=1e6),
        temp=0.1,
        freq0=5e9,
        power=1e-12,
        Qc=2e4,
        qint_init=1e5,
    )

    assert qint == 24.0


def test_qint_vs_temp_consistent_uses_local_seeded_solver(monkeypatch):
    calls = []

    def fake_solve(params, temp, freq0, power, qc, qint_init, fitQP=True):
        calls.append((temp, freq0, power, qc, qint_init, fitQP))
        return qint_init + 1.0

    monkeypatch.setattr("rrfit.waterfall._solve_consistent_qint_local", fake_solve)

    result = QIntVsTemp_consistent(
        [0.1, 0.2],
        _make_params(),
        [5e9, 5.1e9],
        [1e-12, 2e-12],
        2e4,
        [100.0, 200.0],
    )

    assert np.array_equal(result, np.array([101.0, 201.0]))
    assert calls == [
        (0.1, 5e9, 1e-12, 2e4, 100.0, True),
        (0.2, 5.1e9, 2e-12, 2e4, 200.0, True),
    ]


def test_qint_vs_temp_consistent_can_use_fast_solver(monkeypatch):
    calls = []

    def fake_solve(params, temp, freq0, power, qc, qint_init, fitQP=True):
        calls.append((temp, freq0, power, qc, qint_init, fitQP))
        return qint_init + 2.0

    monkeypatch.setattr("rrfit.waterfall._solve_consistent_qint", fake_solve)

    result = QIntVsTemp_consistent(
        [0.1, 0.2],
        _make_params(),
        [5e9, 5.1e9],
        [1e-12, 2e-12],
        2e4,
        [100.0, 200.0],
        inner_solver="fast",
    )

    assert np.array_equal(result, np.array([102.0, 202.0]))
    assert calls == [
        (0.1, 5e9, 1e-12, 2e4, 100.0, True),
        (0.2, 5.1e9, 2e-12, 2e4, 200.0, True),
    ]


def test_qint_vs_temp_consistent_rejects_unknown_inner_solver():
    with pytest.raises(ValueError, match="Unknown inner_solver"):
        QIntVsTemp_consistent(
            [0.1],
            _make_params(),
            [5e9],
            [1e-12],
            2e4,
            [100.0],
            inner_solver="mystery",
        )


def test_fit_waterfall_staged_propagates_fast_inner_solver(monkeypatch):
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
        inner_solver="local",
    ):
        calls.append(("fitIterated", consistent, inner_solver))
        device.best_params = init_params.copy()
        return "search"

    def fake_fit_qint(
        device,
        init_params,
        consistent=False,
        makePlot=True,
        inner_solver="local",
    ):
        calls.append(("Fit_QIntVsTemp", consistent, inner_solver))
        return init_params.copy(), None, None

    monkeypatch.setattr("rrfit.waterfall.fitIterated", fake_fit_iterated)
    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", fake_fit_qint)

    fit_waterfall_staged(
        Device(name="device"),
        {"delta_QP0": (0.0, 5.0)},
        coarse_iterations=1,
        refine_iterations=1,
        makePlot=False,
        init_params=_make_params(),
        inner_solver="fast",
    )

    assert calls == [
        ("fitIterated", False, "fast"),
        ("fitIterated", True, "fast"),
        ("Fit_QIntVsTemp", True, "fast"),
    ]
