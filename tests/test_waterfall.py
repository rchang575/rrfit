from collections import deque

from lmfit import Parameters
import numpy as np

from rrfit.dataio import Device
from rrfit.fitfns import dBmtoW
from rrfit.waterfall import (
    _get_waterfall_arrays,
    _solve_consistent_qint,
    _store_waterfall_fit_summary,
    _solve_consistent_qint_reduced,
    fitIterated,
    fit_waterfall_global,
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

    def fake_fit(device, params, consistent=False, consistent_method="qint", makePlot=True):
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

    def fake_consistent(temp, params, freq0, power, qc, qint_init, consistent_method="qint"):
        calls["count"] += 1
        assert consistent_method == "qint"
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

    def fake_fit_qint(device, init_params, consistent=False, consistent_method="qint", makePlot=True):
        calls.append(("Fit_QIntVsTemp", consistent, consistent_method, init_params["delta_QP0"].value))
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
        ("Fit_QIntVsTemp", True, "qint", 4.0),
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

    def fake_fit(device, params, consistent=False, consistent_method="qint", makePlot=True):
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


def test_solve_consistent_qint_reduced_uses_bracketed_root(monkeypatch):
    calls = []

    class RootResult:
        converged = True
        root = 0.02

    def fake_root_scalar(fn, bracket, method, maxiter):
        calls.append((bracket, method, maxiter, fn(0.02)))
        return RootResult()

    monkeypatch.setattr("rrfit.waterfall.optimize.root_scalar", fake_root_scalar)

    qint = _solve_consistent_qint_reduced(
        _make_params(delta_qp0=1e-6, q_tls0=1e6),
        temp=0.1,
        freq0=5e9,
        power=1e-12,
        Qc=2e4,
        qint_init=1e5,
    )

    assert qint == 50.0
    assert calls[0][1] == "brentq"


def test_fit_waterfall_global_runs_global_seed_then_local_fit(monkeypatch):
    class DEResult:
        x = [3.5]
        fun = 1.25

    calls = []

    def fake_de(func, bounds, args, maxiter, popsize, tol, polish, seed, init):
        calls.append(("de", bounds, maxiter, popsize, tol, polish, seed, init))
        score = func(
            [3.5],
            *args,
        )
        assert np.isfinite(score)
        return DEResult()

    def fake_fit_qint(device, init_params, consistent=False, consistent_method="qint", makePlot=True):
        calls.append(("fit", consistent, consistent_method, init_params["delta_QP0"].value, makePlot))
        fit_params = init_params.copy()
        fit_params["delta_QP0"].value += 1.0
        return fit_params, None, None

    monkeypatch.setattr("rrfit.waterfall.optimize.differential_evolution", fake_de)
    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", fake_fit_qint)

    device = Device(
        name="device",
        line_attenuation=0.0,
        traces=[
            type(
                "Trace",
                (),
                {
                    "is_excluded": False,
                    "power": -30.0,
                    "temperature": 0.1,
                    "fr": 5e9,
                    "Qi": 1e5,
                    "Qi_err": 1e3,
                    "absQc": 2e4,
                    "Ql": 1.67e4,
                },
            )()
        ],
    )

    fit_params, init_fig, fitted_fig = fit_waterfall_global(
        device,
        {"delta_QP0": (0.0, 5.0)},
        init_params=_make_params(delta_qp0=1.0, q_tls0=1e6),
        makePlot=False,
        consistent_method="inverse_qint",
        maxiter=8,
        popsize=6,
        seed=123,
    )

    assert calls == [
        ("de", [(0.0, 5.0)], 8, 6, 0.01, False, 123, "latinhypercube"),
        ("fit", True, "inverse_qint", 3.5, False),
    ]
    assert fit_params["delta_QP0"].value == 4.5
    assert init_fig is None
    assert fitted_fig is None
    assert device.best_params["delta_QP0"].value == 4.5
    assert device.waterfall_global_result.fun == 1.25
