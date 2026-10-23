import contextlib
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from synthetic import PHI0, QC, S21_QI, s21_device, s21_trace
from rrfit.hangerfit import fit_s21_v2
from rrfit.tracefit import (
    _robust_sigma,
    clean_reference,
    fit_magnitude_fallback,
    fit_traces,
    refit_fixed,
)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        out = fn(*args, **kwargs)
    plt.close("all")
    return out


def _fitted(trace, magnitude=False):
    trace.id = 0
    _quiet(fit_s21_v2, trace)
    trace.fit_method = "circle"
    if magnitude:
        assert _quiet(fit_magnitude_fallback, trace)
    return trace


def test_clean_device_triggers_no_refits():
    device = s21_device()

    summary = _quiet(fit_traces, device)

    assert summary["refit"] == summary["drifted"] == []
    assert device.n_clean == 3  # -50, -40, -30 dBm
    assert device.phi_ref == pytest.approx(PHI0, abs=1e-3)
    assert device.absQc_ref == pytest.approx(QC, rel=1e-3)
    for tr in device.traces:
        assert tr.fit_method == tr.fit_method_free == "circle"
        assert tr.drift == ""
        assert (tr.Qi, tr.phi, tr.absQc) == (tr.Qi_free, tr.phi_free, tr.absQc_free)
        assert tr.Qi == pytest.approx(S21_QI, rel=0.05)
        assert (tr.phi_ref, tr.absQc_ref) == (device.phi_ref, device.absQc_ref)


def test_clean_power_min_defaults_to_minus_50_dbm():
    device = s21_device()
    _quiet(fit_traces, device)
    assert device.n_clean == 3
    assert clean_reference(device.traces)["n_clean"] == 3
    assert clean_reference(device.traces, clean_power_min=-60.0)["n_clean"] == 4


def test_reference_sigma_is_floored_at_the_clean_fit_errors():
    device = s21_device()
    for tr in device.traces:
        _fitted(tr)
    clean = device.traces[1:3]
    clean[1].phi, clean[1].absQc = clean[0].phi, clean[0].absQc  # identical: robust spread 0

    ref = clean_reference(clean)

    assert ref["n_clean"] == 2
    assert ref["phi_sigma"] == pytest.approx(np.median([t.phi_err for t in clean]))
    assert ref["absQc_sigma"] == pytest.approx(np.median([t.absQc_err for t in clean]))
    assert ref["phi_sigma"] > 0 and ref["absQc_sigma"] > 0


def test_noisy_low_power_trace_is_refit_with_fixed_absqc():
    device = s21_device([s21_trace(-80.0, 0.2, 0)])

    summary = _quiet(fit_traces, device)

    tr = device.traces[4]
    assert summary["refit"] == [4]
    assert tr.drift == "absQc"
    assert tr.fit_method == "circle_fixed_absQc"
    assert tr.absQc_free < 0.9 * QC
    assert tr.Qi_free < 0.3 * S21_QI
    assert tr.absQc == device.absQc_ref
    assert tr.absQc_err == device.absQc_ref_sigma > 0
    # the stored normalization is ~12% off, beyond the default +/-10% background bound
    assert tr.background_at_bound is True
    assert tr.refit_rejected == ""
    assert abs(tr.Qi - S21_QI) < tr.Qi_err
    assert tr.Qi > 2.5 * tr.Qi_free


def test_wider_background_bound_recovers_noisy_trace():
    device = s21_device([s21_trace(-80.0, 0.2, 0)])

    _quiet(fit_traces, device, background_rtol=0.3)

    tr = device.traces[4]
    assert tr.fit_method == "circle_fixed_absQc"
    assert tr.background_at_bound is False
    assert tr.Qi == pytest.approx(S21_QI, rel=0.25)


def test_noisy_trace_needs_background_refit_to_recover_qi():
    fixed_background = s21_device([s21_trace(-80.0, 0.2, 0)])

    _quiet(fit_traces, fixed_background, refit_background=False)

    tr = fixed_background.traces[4]
    assert tr.fit_method == "circle_fixed_absQc"
    assert tr.Qi < 0.4 * S21_QI
    assert tr.background_amp == tr.background_amp_free


def test_background_tilt_phi_drift_is_refit_with_fixed_phi():
    device = s21_device([s21_trace(-70.0, 0.002, 1, background_slope=-0.1)])

    _quiet(fit_traces, device)

    tr = device.traces[4]
    assert tr.drift == "phi"
    assert tr.fit_method == "circle_fixed_phi"
    assert tr.phi_free - PHI0 > 0.1
    assert tr.phi == device.phi_ref
    assert tr.phi_err == device.phi_ref_sigma > 0
    assert tr.Qi == pytest.approx(S21_QI, rel=0.15)
    assert tr.Qi_free < 0.6 * S21_QI


def _negative_qi_trace():
    # 100x overcoupled with a background tilt: the free circle fit gives Qi < 0.
    return s21_trace(-80.0, 0.002, 0, qi=1e8, background_slope=0.05)


def test_negative_qi_circle_fixed_refits_both_parameters():
    device = s21_device([_negative_qi_trace()])

    summary = _quiet(fit_traces, device, negative_qi="circle_fixed")

    tr = device.traces[4]
    assert summary["refit"] == [4]
    assert tr.fit_method_free == "circle" and tr.Qi_free < 0
    assert "Qi<=0" in tr.drift.split(",")
    assert tr.fit_method == "circle_fixed_phi_absQc"
    assert (tr.phi, tr.absQc) == (device.phi_ref, device.absQc_ref)
    assert tr.Qi == pytest.approx(1e8, rel=0.1)


def test_negative_qi_defaults_to_magnitude_fallback():
    device = s21_device([_negative_qi_trace()])

    _quiet(fit_traces, device)

    tr = device.traces[4]
    assert tr.fit_method_free == "magnitude"
    assert tr.Qi_free > 0
    assert tr.fit_method.startswith("magnitude")


def test_negative_qi_circle_fixed_falls_back_to_magnitude_when_refit_fails(monkeypatch):
    def failing_refit(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr("rrfit.tracefit._refit_circle_fixed", failing_refit)
    device = s21_device([_negative_qi_trace()])

    with pytest.warns(RuntimeWarning, match="trace#4 rejected"):
        summary = _quiet(fit_traces, device, negative_qi="circle_fixed")

    assert summary["refit"] == []
    assert summary["rejected"] == {4: "error"}
    assert device.traces[4].refit_rejected == "error"
    assert device.traces[4].fit_method == "magnitude"
    assert device.traces[4].Qi > 0


def test_drifted_magnitude_trace_is_refit_with_magnitude_model():
    # 100x overcoupled: the circle fit goes negative, the magnitude fallback's phi drifts.
    device = s21_device([s21_trace(-80.0, 0.05, 1, qi=1e8, background_slope=0.1)])

    _quiet(fit_traces, device)

    tr = device.traces[4]
    assert tr.fit_method_free == "magnitude"
    assert tr.drift == "phi"
    assert tr.fit_method == "magnitude_fixed_phi"
    assert tr.phi == device.phi_ref


@pytest.mark.parametrize(
    "kwargs, fit_method",
    [
        (dict(phi=(PHI0, 0.005)), "magnitude_fixed_phi"),
        (dict(absQc=(QC, 5e3)), "magnitude_fixed_absQc"),
        (dict(phi=(PHI0, 0.005), absQc=(QC, 5e3)), "magnitude_fixed_phi_absQc"),
    ],
)
def test_refit_fixed_magnitude_path_recovers_qi(kwargs, fit_method):
    tr = _fitted(s21_trace(-50.0, 0.002, 3, background_slope=-0.05), magnitude=True)
    assert abs(tr.phi - PHI0) > 0.05

    assert _quiet(refit_fixed, tr, **kwargs)

    assert tr.fit_method == fit_method
    assert tr.Qi == pytest.approx(S21_QI, rel=0.05)
    if "phi" in kwargs:
        assert (tr.phi, tr.phi_err) == kwargs["phi"]
    if "absQc" in kwargs:
        assert (tr.absQc, tr.absQc_err) == kwargs["absQc"]


def test_refit_fixed_propagates_reference_sigma_into_qi_err():
    narrow = _fitted(s21_trace(-50.0, 0.002, 3))
    wide = _fitted(s21_trace(-50.0, 0.002, 3))

    _quiet(refit_fixed, narrow, absQc=(QC, 1.0))
    _quiet(refit_fixed, wide, absQc=(QC, 1e4))

    assert narrow.Qi == pytest.approx(wide.Qi, rel=1e-6)
    assert wide.Qi_err > 5 * narrow.Qi_err
    # d(1/Qi) = cos(phi) / absQc**2 * d(absQc) dominates the wide case
    expected = wide.Qi**2 * np.cos(PHI0) / QC**2 * 1e4
    assert wide.Qi_err == pytest.approx(expected, rel=0.2)


def test_refit_fixed_rejects_nonpositive_qi_and_keeps_free_fit(monkeypatch):
    from types import SimpleNamespace

    from uncertainties import ufloat

    def negative_refit(trace, phi_u=None, absQc_u=None, refit_background=True, background_rtol=0.1):
        u = ufloat(1.0, 0.1)
        values = dict(fr=u, Ql=u, absQc=u, phi=u, Qi=ufloat(-1e7, 1e6))
        return SimpleNamespace(chisqr=1.0, redchi=1.0), values, {"background_amp": 2.0}

    monkeypatch.setattr("rrfit.tracefit._refit_circle_fixed", negative_refit)
    tr = _fitted(s21_trace(-50.0, 0.002, 3))
    before = dict(vars(tr))

    with pytest.warns(RuntimeWarning, match=r"Qi = -1e\+07.*keeping the free fit"):
        assert not refit_fixed(tr, absQc=(QC, 1e3))

    assert tr.refit_rejected == "Qi<=0"
    del tr.refit_rejected
    assert vars(tr).keys() == before.keys()
    for key, value in before.items():
        assert vars(tr)[key] is value


def test_refit_fixed_requires_a_parameter():
    with pytest.raises(ValueError, match="phi and/or absQc"):
        refit_fixed(s21_trace(-50.0, 0.002, 3))


def test_too_few_clean_traces_warns_and_keeps_free_fits():
    traces = [s21_trace(p, 0.002, i) for i, p in enumerate((-50.0, -60.0, -80.0))]
    traces.append(s21_trace(-80.0, 0.2, 0))
    device = s21_device()
    device.traces = traces
    for idx, tr in enumerate(traces):
        tr.id = idx

    with pytest.warns(RuntimeWarning, match=r"only 1 clean traces \(min_clean=2\)"):
        summary = _quiet(fit_traces, device)

    assert summary == {"refit": [], "drifted": [], "rejected": {}}
    assert device.phi_ref is None and device.n_clean == 0
    assert all(tr.fit_method == "circle" for tr in traces)
    assert all(tr.drift == "" for tr in traces)


def test_auto_fix_false_only_flags_drift():
    device = s21_device([s21_trace(-80.0, 0.2, 0)])

    summary = _quiet(fit_traces, device, auto_fix=False)

    tr = device.traces[4]
    assert summary["drifted"] == [4] and summary["refit"] == []
    assert tr.drift == "absQc"
    assert tr.fit_method == "circle"
    assert tr.Qi == tr.Qi_free


def test_fit_traces_accepts_a_list_of_traces():
    device = s21_device([s21_trace(-80.0, 0.2, 0)])

    summary = _quiet(fit_traces, device.traces)

    assert summary["refit"] == [4]
    assert not hasattr(device, "phi_ref")


def test_fit_traces_rejects_unknown_negative_qi_option():
    with pytest.raises(ValueError, match="negative_qi"):
        fit_traces(s21_device(), negative_qi="drop")


def test_clean_reference_selection_and_robust_spread():
    device = s21_device()
    for tr in device.traces:
        _fitted(tr)
    device.traces[0].fit_method = "magnitude"
    device.traces[1].is_excluded = True
    device.traces[2].phi_err = 0.0

    assert clean_reference(device.traces) is None
    device.traces[2].phi_err = 1e-3
    assert clean_reference(device.traces)["n_clean"] == 2


def test_robust_sigma():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert _robust_sigma(values) == pytest.approx(1.4826)
    assert _robust_sigma(np.array([1.0, 1.0, 1.0, 2.0])) == pytest.approx(0.5)


@pytest.mark.parametrize("plot", [False, True])
def test_fit_s21_v2_only_draws_figures_when_plotting(plot):
    plt.close("all")
    trace = s21_trace(-50.0, 0.002, 3)

    with contextlib.redirect_stdout(io.StringIO()):
        fit_s21_v2(trace, plot=plot)

    assert bool(plt.get_fignums()) is plot
    assert trace.Qi == pytest.approx(S21_QI, rel=0.05)
    plt.close("all")


def _device_from(traces):
    device = s21_device()
    device.traces = traces
    for idx, tr in enumerate(traces):
        tr.id = idx
    return device


def test_refit_rejected_when_redchi_gets_much_worse():
    device = s21_device([s21_trace(-80.0, 0.2, 0)])

    with pytest.warns(RuntimeWarning, match=r"trace#4 rejected \(redchi .* > 0\.5x\)"):
        summary = _quiet(fit_traces, device, redchi_ratio_max=0.5)

    tr = device.traces[4]
    assert summary["refit"] == [] and summary["rejected"] == {4: "redchi"}
    assert tr.refit_rejected == "redchi"
    assert tr.fit_method == "circle"
    assert (tr.Qi, tr.absQc, tr.redchi) == (tr.Qi_free, tr.absQc_free, tr.redchi_free)
    assert tr.background_amp == tr.background_amp_free


def test_refit_fixed_redchi_guard_can_be_disabled():
    tr = _fitted(s21_trace(-80.0, 0.2, 0))

    assert _quiet(refit_fixed, tr, absQc=(QC, 1e3), redchi_ratio_max=None)
    assert tr.refit_rejected == ""


def _undercoupled_device(qi=2e6):
    # Qi / |Qc| = 2 < overcoupling_min; the tilted -70 dBm trace drifts in phi
    traces = [s21_trace(p, 0.002, i, qi=qi) for i, p in enumerate((-60.0, -50.0, -40.0, -30.0))]
    traces.append(s21_trace(-70.0, 0.002, 1, qi=qi, background_slope=-0.2))
    return _device_from(traces)


def test_undercoupled_drift_is_flagged_but_not_refit():
    device = _undercoupled_device()

    summary = _quiet(fit_traces, device)

    tr = device.traces[4]
    assert "phi" in tr.drift.split(",")
    assert summary["drifted"] == [4] and summary["refit"] == []
    assert summary["rejected"] == {4: "undercoupled"}
    assert tr.refit_rejected == "undercoupled"
    assert tr.fit_method == "circle" and tr.Qi == tr.Qi_free


def test_overcoupling_min_controls_the_gate():
    device = _undercoupled_device()

    _quiet(fit_traces, device, overcoupling_min=1.0)

    assert device.traces[4].fit_method.startswith("circle_fixed")


def test_fix_phi_overcoupled_refits_undetected_phi_offset():
    # a 0.035 rad phi offset is below phi_tol but biases Qi on a 20x overcoupled trace
    default = s21_device([s21_trace(-70.0, 0.002, 1, background_slope=-0.03)])
    _quiet(fit_traces, default)
    assert default.traces[4].drift == ""
    assert default.traces[4].fit_method == "circle"
    assert default.traces[4].Qi < 0.85 * S21_QI

    device = s21_device([s21_trace(-70.0, 0.002, 1, background_slope=-0.03)])
    summary = _quiet(fit_traces, device, fix_phi_overcoupled=True)

    tr = device.traces[4]
    # trace 0 (-60 dBm) is overcoupled but not a reference trace, so it is refit too
    assert summary["refit"] == [0, 4] and summary["drifted"] == []
    assert tr.fit_method == "circle_fixed_phi"
    assert (tr.phi, tr.phi_err) == (device.phi_ref, device.phi_ref_sigma)
    assert tr.Qi == pytest.approx(S21_QI, rel=0.05)
    assert device.traces[0].Qi == pytest.approx(S21_QI, rel=0.05)
    # reference traces are left alone
    assert all(t.fit_method == "circle" for t in device.traces[1:4])


def test_fix_phi_overcoupled_skips_undercoupled_traces():
    device = _undercoupled_device()
    device.traces[4] = s21_trace(-70.0, 0.002, 1, qi=2e6, background_slope=-0.03)
    device.traces[4].id = 4

    summary = _quiet(fit_traces, device, fix_phi_overcoupled=True)

    assert summary["refit"] == []
    assert device.traces[4].fit_method == "circle"
