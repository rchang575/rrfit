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


def test_fitIterated_raises_chained_error_after_retries(monkeypatch):
    def failing_fit(device, params, consistent=False, makePlot=True, inner_solver="local"):
        raise ValueError("NaN in residual")

    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", failing_fit)

    with pytest.raises(RuntimeError, match="exceeded retries=3") as excinfo:
        fitIterated(
            Device(name="device"),
            {"delta_QP0": (0.0, 5.0)},
            numIter=2,
            makePlot=False,
            retries=3,
            init_params=_make_params(),
        )

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "NaN in residual" in str(excinfo.value)


def _histogram_inputs(nfree):
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace
    from rrfit.waterfall import createFitHistograms

    device = Device(name="device")
    device.waterfall_fit_result = SimpleNamespace(nfree=nfree)
    names = ["delta_QP0", "Q_TLS0", "tc", "Q_other", "beta", "beta2", "D_0"]
    bounds = {name: (0.0, 1.0) for name in names}
    init = {name: np.linspace(0.1, 0.9, 5) for name in names}
    final = {name: np.linspace(0.2, 0.8, 5) for name in names}
    red_chi2 = np.array([1.0, 1.01, 1.02, 5.0, 50.0])
    return createFitHistograms, device, init, final, bounds, red_chi2


@pytest.mark.parametrize("nfree", [0, -1])
def test_createFitHistograms_skips_when_no_degrees_of_freedom(nfree):
    createFitHistograms, device, init, final, bounds, red_chi2 = _histogram_inputs(nfree)

    with pytest.warns(RuntimeWarning, match="nfree"):
        figs = createFitHistograms(device, init, final, bounds, red_chi2)

    assert figs == (None, None, None)


def test_createFitHistograms_still_plots_when_well_posed():
    import matplotlib.pyplot as plt

    createFitHistograms, device, init, final, bounds, red_chi2 = _histogram_inputs(10)

    figs = createFitHistograms(device, init, final, bounds, red_chi2)

    assert all(fig is not None for fig in figs)
    plt.close("all")


def test_fitIterated_with_plots_completes_when_no_degrees_of_freedom(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace

    def fake_fit(device, params, consistent=False, makePlot=True, inner_solver="local"):
        device.waterfall_fit_result = SimpleNamespace(nfree=0)
        if makePlot:
            return params.copy(), None, None
        return params.copy(), 1.0

    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", fake_fit)
    bounds = {name: (0.5, 1.5) for name in _make_params().keys()}

    with pytest.warns(RuntimeWarning, match="nfree"):
        result = fitIterated(
            Device(name="device"), bounds, numIter=2, makePlot=True,
            init_params=_make_params(),
        )

    init_dict, final_dict, red_chi2, figures = result
    assert figures[:3] == [None, None, None]


# Parameters and trace values from a real fit where the plotted -40 dBm curve
# (data only at 100 mK) collapsed onto the local solver's qint_init/50 floor
# below ~35 mK.
_LOW_T_DROP_PARAMS = dict(
    delta_QP0=4.994485263184948e-05,
    Q_TLS0=22181720.416532505,
    tc=0.1309261452687087,
    Q_other=59008618.884217545,
    beta=6.03677456049002e-16,
    beta2=1.9999999999999998,
    D_0=356.12166734157006,
)
_LOW_T_DROP_TRACE = dict(fr=4652943848.8096075, Qi=412123.99565669073, absQc=1452963.6851049897)


def _low_t_drop_params():
    params = Parameters()
    for name, value in _LOW_T_DROP_PARAMS.items():
        params.add(name, value=value)
    return params


def _relative_residual(qint, params, temp, freq0, power, qc):
    from rrfit.waterfall import _consistent_qint_residual

    return abs(_consistent_qint_residual(qint, params, temp, freq0, power, qc)) / qint


def test_consistent_curve_continuation_tracks_root_far_from_single_data_point():
    params = _low_t_drop_params()
    temp_axis = np.linspace(0.0099843125, 0.10002033333333334, 30)
    fr, qi, qc = (_LOW_T_DROP_TRACE[key] for key in ("fr", "Qi", "absQc"))
    freq0 = np.full(temp_axis.size, fr)
    power = np.full(temp_axis.size, dBmtoW(-40.0 - 80.0))
    seeds = np.full(temp_axis.size, qi)

    seeded = predict_qint_consistent_curve(temp_axis, params, freq0, power, qc, seeds)
    continued = predict_qint_consistent_curve(
        temp_axis, params, freq0, power, qc, seeds, anchor_temps=[0.099895]
    )

    # Old behaviour (kept when anchor_temps is not given): stuck on the floor at low T.
    assert seeded[0] == pytest.approx(qi / 50, rel=1e-3)
    # Continuation lands on a true self-consistent root everywhere.
    for t, q in zip(temp_axis, continued):
        assert _relative_residual(q, params, t, fr, power[0], qc) < 1e-6
    assert continued[0] > 5e7
    # Q decreases monotonically with T for these parameters: no vertical drop.
    assert np.all(np.diff(continued) < 0)
    # Near the measured point the curve is unchanged.
    assert continued[-1] == pytest.approx(seeded[-1], rel=1e-6)


def test_consistent_curve_continuation_matches_seeded_curve_inside_data_range():
    params = _low_t_drop_params()
    fr, qc = _LOW_T_DROP_TRACE["fr"], _LOW_T_DROP_TRACE["absQc"]
    data_temps = np.array([0.04, 0.06, 0.08, 0.1])
    power_w = dBmtoW(-40.0 - 80.0)
    from rrfit.waterfall import _solve_consistent_qint

    data_qi = np.array(
        [_solve_consistent_qint(params, t, fr, power_w, qc, 1e6) for t in data_temps]
    )
    temp_axis = np.linspace(0.01, 0.1, 25)
    seeds = np.interp(temp_axis, data_temps, data_qi)
    freq0 = np.full(temp_axis.size, fr)
    power = np.full(temp_axis.size, power_w)

    seeded = predict_qint_consistent_curve(temp_axis, params, freq0, power, qc, seeds)
    continued = predict_qint_consistent_curve(
        temp_axis, params, freq0, power, qc, seeds, anchor_temps=data_temps
    )

    inside = temp_axis >= data_temps.min()
    np.testing.assert_array_equal(continued[inside], seeded[inside])


def test_plot_Qi_vs_temp_curve_has_no_low_temperature_drop():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rrfit.dataio import Trace
    from rrfit.waterfall import plot_Qi_vs_temp

    fr, qi, qc = (_LOW_T_DROP_TRACE[key] for key in ("fr", "Qi", "absQc"))

    def trace(power, temperature, qi):
        return Trace(
            power=power, temperature=temperature, fr=fr, Qi=qi, Qi_err=0.01 * qi,
            Ql=1 / (1 / qi + 1 / qc), absQc=qc, is_excluded=False,
        )

    # -40 dBm is measured only at 100 mK; -80 dBm sets the 10 mK end of the axis.
    device = Device(
        name="device",
        traces=[trace(-40.0, 0.099895, qi), trace(-80.0, 0.0099843125, 5e6),
                trace(-80.0, 0.10002033333333334, 4e5)],
        line_attenuation=80.0,
    )

    fig, ax = plot_Qi_vs_temp(device, plotParams=_low_t_drop_params())

    curve = next(line for line in ax.get_lines() if line.get_label() == "-40.0 dBm")
    ys = curve.get_ydata()
    assert ys.min() > 1e5
    assert np.all(np.diff(ys) < 0)
    plt.close(fig)


def _quiet(fn, *args, **kwargs):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def test_fit_qintvstemp_loss_space_requires_loss_errors():
    from synthetic import overcoupled_waterfall_device, waterfall_params
    from rrfit.waterfall import Fit_QIntVsTemp

    with pytest.raises(ValueError, match="compute_loss_errors"):
        Fit_QIntVsTemp(
            overcoupled_waterfall_device(), waterfall_params(),
            makePlot=False, loss_space=True,
        )


def test_fit_qintvstemp_loss_space_fixed_nbar_residuals_are_in_loss():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from synthetic import overcoupled_waterfall_device, waterfall_params
    from rrfit.fitfns import nbarvsPin
    from rrfit.losserr import compute_loss_errors
    from rrfit.waterfall import Fit_QIntVsTemp, QIntVsTemp_TLS_QP_Beta_fit_usingParams

    device = overcoupled_waterfall_device()
    compute_loss_errors(device)

    params, _init_fig, fitted_fig = _quiet(
        Fit_QIntVsTemp, device, waterfall_params(Q_TLS0=2e7, tc=1.1), loss_space=True
    )

    arrays = _get_waterfall_arrays(device)
    nbar = nbarvsPin(arrays["power_watts"], arrays["freq0"], arrays["ql"], arrays["qc"])
    q_model = QIntVsTemp_TLS_QP_Beta_fit_usingParams(
        arrays["temperature"], params, arrays["freq0"], nbar, 0
    )
    loss_err = np.array([tr.loss_err for tr in arrays["traces"]])
    np.testing.assert_allclose(
        device.waterfall_fit_result.residual,
        (1 / arrays["qint"] - 1 / q_model) / loss_err,
        rtol=1e-10,
    )
    assert fitted_fig is not None
    plt.close("all")


def test_fit_qintvstemp_loss_space_consistent_on_overcoupled_device():
    import copy

    from synthetic import WATERFALL_TRUE, overcoupled_waterfall_device, waterfall_params
    from rrfit.losserr import compute_loss_errors
    from rrfit.waterfall import Fit_QIntVsTemp

    device = overcoupled_waterfall_device(seed=0)
    compute_loss_errors(device)
    default = copy.deepcopy(device)
    init = waterfall_params(Q_TLS0=2e7, tc=1.1)

    _quiet(Fit_QIntVsTemp, default, init.copy(), consistent=True, makePlot=False)
    params, red_chi2 = _quiet(
        Fit_QIntVsTemp, device, init.copy(), consistent=True, makePlot=False, loss_space=True
    )

    assert default.waterfall_fit_result.redchi > 100
    assert red_chi2 < 3
    for name in ("Q_TLS0", "Q_other", "tc"):
        assert abs(params[name].value - WATERFALL_TRUE[name]) < 3 * params[name].stderr
    assert set(device.waterfall_best_values) == set(default.waterfall_best_values)


def _record_fit_calls(monkeypatch):
    calls = []

    def fake_fit(device, params, consistent=False, makePlot=True, inner_solver="local", **kwargs):
        calls.append(kwargs)
        if makePlot:
            return params.copy(), None, None
        return params.copy(), 1.0

    monkeypatch.setattr("rrfit.waterfall.Fit_QIntVsTemp", fake_fit)
    return calls


@pytest.mark.parametrize("loss_space", [False, True])
def test_fitIterated_forwards_loss_space_only_when_on(monkeypatch, loss_space):
    calls = _record_fit_calls(monkeypatch)

    fitIterated(
        Device(name="device"), {"delta_QP0": (0.0, 1.0)}, numIter=2,
        makePlot=False, init_params=_make_params(), loss_space=loss_space,
    )

    expected = {"loss_space": True} if loss_space else {}
    assert len(calls) == 3
    assert all(kwargs == expected for kwargs in calls)


@pytest.mark.parametrize("loss_space", [False, True])
def test_fit_waterfall_staged_forwards_loss_space_only_when_on(monkeypatch, loss_space):
    calls = _record_fit_calls(monkeypatch)

    fit_waterfall_staged(
        Device(name="device"), {"delta_QP0": (0.0, 1.0)}, coarse_iterations=2,
        refine_iterations=1, makePlot=True, init_params=_make_params(),
        loss_space=loss_space,
    )

    expected = {"loss_space": True} if loss_space else {}
    assert len(calls) == 3 + 2 + 1
    assert all(kwargs == expected for kwargs in calls)
