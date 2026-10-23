"""Device-level, drift-aware S21 fitting.

The complex coupling Qc (``absQc`` and ``phi``) is set by geometry and should
not depend on power or temperature. On noisy low-power traces the circle fit
distorts the circle, so ``absQc`` and ``phi`` drift, and for overcoupled
resonators ``1/Qi = 1/Ql - cos(phi)/absQc`` absorbs that error and can even go
negative. ``fit_traces`` fits every trace freely, takes reference values of
``phi`` and ``absQc`` from the device's clean traces, and refits the traces
whose ``phi`` or ``absQc`` drifted (or whose ``Qi`` is not positive) with those
parameters fixed to the reference.
"""

import warnings
from typing import Optional, Union

import numpy as np
from uncertainties import ufloat
from uncertainties.umath import atan2 as uatan2, cos as ucos, sin as usin, sqrt as usqrt

from rrfit.dataio import Device, Trace
from rrfit.hangerfit import fit_s21_v2
from rrfit.models import S21LogMagModel, S21Model

# trace attributes written by a fit (free or fixed)
_FIT_ATTRS = (
    "fr", "fr_err", "Ql", "Ql_err", "absQc", "absQc_err", "phi", "phi_err",
    "Qi", "Qi_err", "chisqr", "redchi",
)
# free-fit values kept on every trace as ``<name>_free``
_FREE_ATTRS = (
    "Qi", "Qi_err", "phi", "phi_err", "absQc", "absQc_err", "Ql", "redchi", "fit_method",
    "background_amp", "background_phase",
)


def _store(trace: Trace, **values):
    for name, value in values.items():
        setattr(trace, name, value)


def _store_uvalues(trace: Trace, fr, Ql, absQc, phi, Qi):
    _store(
        trace,
        fr=fr.n, fr_err=fr.s, Ql=Ql.n, Ql_err=Ql.s, absQc=absQc.n, absQc_err=absQc.s,
        phi=phi.n, phi_err=phi.s, Qi=Qi.n, Qi_err=Qi.s,
    )


def _temp_mk(trace: Trace) -> float:
    temp = trace.temperature
    return np.nan if temp is None or np.isnan(temp) else temp * 1e3


def _s21_db(trace: Trace) -> np.ndarray:
    return 20 * np.log10(np.abs(trace.s21real + 1j * trace.s21imag))


def fit_magnitude_fallback(trace: Trace) -> bool:
    """Replace a trace's circle fit with a fit of ``|S21|`` (dB) to ``S21LogMagModel``.

    ``|S21|`` is unchanged under ``(1 - c) -> -(1 - c)``, i.e. ``Qi -> -Qi``, with
    ``c = height * cos(phi)``; the physical branch ``c <= 1`` is taken. Sets
    ``trace.fit_method = "magnitude"`` on success.

    Returns:
        False (trace untouched) if the fit has no uncertainties, else True.
    """
    result = S21LogMagModel().fit(_s21_db(trace), trace.frequency, method="least_squares")
    u = result.uvars
    if u is None:
        print(f"  magnitude fallback failed for trace#{trace.id}: no uncertainties")
        return False
    Ql = u["fr"] / u["fwhm"]
    c = u["height"] * ucos(u["phi"])
    s = u["height"] * usin(u["phi"])
    if c.n > 1:
        c = 2 - c
    phi = uatan2(s, c)
    absQc = Ql / usqrt(c**2 + s**2)
    Qi = Ql / (1 - c)
    print(
        f"  circle fit gave Qi = {trace.Qi:.3g}; magnitude fallback Qi = {Qi.n:.3g} "
        f"+/- {Qi.s:.2g}, Ql = {Ql.n:.3g}, |Qc| = {absQc.n:.3g}"
    )
    _store_uvalues(trace, u["fr"], Ql, absQc, phi, Qi)
    _store(trace, chisqr=result.chisqr, redchi=result.redchi, fit_method="magnitude")
    return True


def _refit_circle_fixed(
    trace: Trace, phi_u=None, absQc_u=None, refit_background=True, background_rtol=0.1
):
    """Circle-model refit with ``phi`` and/or ``absQc`` fixed, seeded from the free fit.

    Rebuilds the normalized S21 from ``tau`` and the stored background, as
    ``fit_s21_v2`` fitted it. Fixed parameters enter ``Qi`` as the given ufloats.
    With ``refit_background``, the background amplitude ``a`` and phase
    ``alpha`` relative to the stored background are fitted too; on noisy traces
    the stored normalization is itself biased, and with it held fixed the
    error moves into ``Ql`` instead. ``a`` is bounded to ``1 +/- background_rtol``
    and ``alpha`` to ``+/- background_rtol`` rad.
    """
    f = trace.frequency
    orp = trace.background_amp * np.exp(1j * trace.background_phase)
    s21 = (trace.s21real + 1j * trace.s21imag) * np.exp(-1j * 2 * np.pi * f * trace.tau) / orp
    model = S21Model()
    params = model.make_params(
        fr={"value": trace.fr, "min": f[0], "max": f[-1]},
        Ql={"value": trace.Ql, "min": 0},
        absQc={"value": trace.absQc if absQc_u is None else absQc_u.n, "min": 0,
               "vary": absQc_u is None},
        phi={"value": trace.phi if phi_u is None else phi_u.n, "min": -np.pi, "max": np.pi,
             "vary": phi_u is None},
        a={"value": 1, "min": 1 - background_rtol, "max": 1 + background_rtol,
           "vary": refit_background},
        alpha={"value": 0, "min": -background_rtol, "max": background_rtol,
               "vary": refit_background},
        tau={"value": 0, "vary": False},
    )
    result = model.fit(s21, f, params=params, method="least_squares")
    u = result.uvars
    fr, Ql = u["fr"], u["Ql"]
    phi = u["phi"] if phi_u is None else phi_u
    absQc = u["absQc"] if absQc_u is None else absQc_u
    Qi = 1 / ((1 / Ql) - (ucos(phi) / absQc))
    background = {}
    if refit_background:
        a, alpha = result.params["a"].value, result.params["alpha"].value
        tol = 1e-3 * background_rtol
        at_bound = abs(abs(a - 1) - background_rtol) < tol or abs(abs(alpha) - background_rtol) < tol
        # model background is a * exp(-1j * alpha) on top of the stored one
        orp *= a * np.exp(-1j * alpha)
        background = dict(
            background_amp=abs(orp), background_phase=np.angle(orp), background_at_bound=at_bound
        )
    return result, dict(fr=fr, Ql=Ql, absQc=absQc, phi=phi, Qi=Qi), background


def _refit_magnitude_fixed(
    trace: Trace, phi_u=None, absQc_u=None, refit_background=True, background_rtol=0.1
):
    """Magnitude-model refit with ``phi`` and/or ``absQc`` fixed, seeded from the free fit.

    The magnitude model has no ``absQc`` parameter, so a fixed ``absQc_fixed``
    is added and ``height = fr / (fwhm * absQc_fixed)`` (= Ql/absQc). The fixed
    values select the branch, so no ``c > 1`` flip is applied. The dB offset
    ``ofs`` is always fitted, so ``refit_background`` and ``background_rtol``
    have no effect here.
    """
    f = trace.frequency
    s21_db = _s21_db(trace)
    model = S21LogMagModel()
    params = model.guess(s21_db, f)
    params["fr"].set(value=trace.fr)
    params["fwhm"].set(value=trace.fr / trace.Ql)
    if phi_u is None:
        params["phi"].set(value=trace.phi)
    else:
        params["phi"].set(value=phi_u.n, vary=False)
    if absQc_u is None:
        params["height"].set(value=trace.Ql / trace.absQc)
    else:
        params.add("absQc_fixed", value=absQc_u.n, vary=False)
        params["height"].set(expr="fr / (fwhm * absQc_fixed)", min=-np.inf, max=np.inf)
    result = model.fit(s21_db, f, params=params, method="least_squares")
    u = result.uvars
    fr = u["fr"]
    Ql = fr / u["fwhm"]
    phi = u["phi"] if phi_u is None else phi_u
    absQc = Ql / u["height"] if absQc_u is None else absQc_u
    Qi = 1 / ((1 / Ql) - (ucos(phi) / absQc))
    return result, dict(fr=fr, Ql=Ql, absQc=absQc, phi=phi, Qi=Qi), {}


def refit_fixed(
    trace: Trace,
    phi=None,
    absQc=None,
    refit_background: bool = True,
    background_rtol: float = 0.1,
    redchi_ratio_max: Optional[float] = 2.0,
) -> bool:
    """Refit ``trace`` with ``phi`` and/or ``absQc`` fixed.

    Uses the circle model for circle fits and the magnitude model for magnitude
    fits (by ``trace.fit_method``). ``phi`` and ``absQc`` are ``(value, sigma)``
    pairs; ``sigma`` is propagated into ``Qi_err`` and stored as ``phi_err`` /
    ``absQc_err``. Sets ``fit_method`` to e.g. ``"circle_fixed_phi_absQc"``.

    Args:
        refit_background: circle path only; also fit the background amplitude
            ``a`` (within ``1 +/- background_rtol``) and phase ``alpha`` (within
            ``+/- background_rtol`` rad) relative to the stored background, and
            update ``background_amp`` / ``background_phase``.
        background_rtol: see ``refit_background``.
        redchi_ratio_max: reject the refit if its ``redchi`` exceeds this
            multiple of the current (free) ``redchi``. None disables the check.

    Sets ``trace.refit_rejected`` to ``""`` on success, else to the reason:
    ``"error"`` (fit failed), ``"Qi<=0"`` (non-positive or non-finite ``Qi``) or
    ``"redchi"``. A circle refit with ``refit_background`` also sets
    ``trace.background_at_bound`` (True if ``a`` or ``alpha`` hit its bound).

    Returns:
        True if the trace was updated. A rejected refit warns and leaves every fit
        attribute of the trace as it was.
    """
    if phi is None and absQc is None:
        raise ValueError("refit_fixed needs phi and/or absQc")
    phi_u = None if phi is None else ufloat(*phi)
    absQc_u = None if absQc is None else ufloat(*absQc)
    base = "magnitude" if getattr(trace, "fit_method", "circle").startswith("magnitude") else "circle"
    refit = _refit_magnitude_fixed if base == "magnitude" else _refit_circle_fixed
    redchi_free = trace.redchi
    try:
        result, values, background = refit(
            trace, phi_u=phi_u, absQc_u=absQc_u, refit_background=refit_background,
            background_rtol=background_rtol,
        )
    except (TypeError, ValueError, ZeroDivisionError) as err:
        reason, detail, background = "error", repr(err), {}
    else:
        qi = values["Qi"]
        if not (np.isfinite(qi.n) and np.isfinite(qi.s) and qi.n > 0):
            reason, detail = "Qi<=0", f"Qi = {qi.n:.3g}"
        elif redchi_ratio_max is not None and result.redchi > redchi_ratio_max * redchi_free:
            reason = "redchi"
            detail = f"redchi {redchi_free:.2e} -> {result.redchi:.2e} > {redchi_ratio_max:g}x"
        else:
            reason = detail = ""
    if "background_at_bound" in background:
        trace.background_at_bound = background.pop("background_at_bound")
    trace.refit_rejected = reason
    if reason:
        warnings.warn(
            f"fixed-parameter refit of trace#{trace.id} rejected ({detail}); keeping the free fit",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    _store_uvalues(trace, **values)
    _store(trace, **background)
    fixed = "_".join(name for name, u in (("phi", phi_u), ("absQc", absQc_u)) if u is not None)
    _store(trace, chisqr=result.chisqr, redchi=result.redchi, fit_method=f"{base}_fixed_{fixed}")
    return True


def _robust_sigma(values: np.ndarray) -> float:
    """1.4826 * MAD, or the sample std if the MAD is zero."""
    sigma = 1.4826 * np.median(np.abs(values - np.median(values)))
    if sigma == 0:
        sigma = np.std(values, ddof=1)
    return float(sigma)


def _is_clean(trace: Trace, clean_power_min: float) -> bool:
    if getattr(trace, "is_excluded", False):
        return False
    if trace.power is None or not trace.power >= clean_power_min:
        return False
    if getattr(trace, "fit_method", None) != "circle":
        return False
    if not (trace.Qi is not None and trace.Qi > 0):
        return False
    errs = (trace.phi_err, trace.absQc_err)
    return all(err is not None and np.isfinite(err) and err != 0 for err in errs)


def clean_reference(traces, clean_power_min: float = -50.0) -> Optional[dict]:
    """Median ``phi`` and ``absQc`` of the clean traces, with their spreads.

    Each spread is ``max(_robust_sigma(values), median of the clean traces' own
    fit errors)``, so it is never smaller than a single trace's fit error, even
    for two traces that happen to agree.

    Clean traces are not excluded, have ``power >= clean_power_min``, a free
    circle fit (``fit_method == "circle"``), ``Qi > 0``, and finite, non-zero
    ``phi_err`` and ``absQc_err``.

    Returns:
        ``{"phi_ref", "phi_sigma", "absQc_ref", "absQc_sigma", "n_clean"}``, or
        None if there are fewer than two clean traces.
    """
    clean = [tr for tr in traces if _is_clean(tr, clean_power_min)]
    if len(clean) < 2:
        return None
    def spread(values, errs):
        return max(_robust_sigma(values), float(np.median(errs)))

    phis = np.array([tr.phi for tr in clean], dtype=float)
    absqcs = np.array([tr.absQc for tr in clean], dtype=float)
    return dict(
        phi_ref=float(np.median(phis)),
        phi_sigma=spread(phis, [tr.phi_err for tr in clean]),
        absQc_ref=float(np.median(absqcs)),
        absQc_sigma=spread(absqcs, [tr.absQc_err for tr in clean]),
        n_clean=len(clean),
    )


def _drift(trace: Trace, ref: dict, phi_tol: float, absqc_rtol: float) -> list[str]:
    drift = []
    if abs(trace.phi - ref["phi_ref"]) > phi_tol:
        drift.append("phi")
    if abs(trace.absQc - ref["absQc_ref"]) / ref["absQc_ref"] > absqc_rtol:
        drift.append("absQc")
    if not trace.Qi > 0:
        drift.append("Qi<=0")
    return drift


def fit_traces(
    device: Union[Device, list],
    *,
    plot: bool = False,
    xpts: int = 5,
    phase_err_threshold: float = 0.01,
    clean_power_min: float = -50.0,
    phi_tol: float = 0.1,
    absqc_rtol: float = 0.1,
    min_clean: int = 2,
    auto_fix: bool = True,
    refit_background: bool = True,
    negative_qi: str = "magnitude",
    redchi_ratio_max: Optional[float] = 2.0,
    background_rtol: float = 0.1,
    overcoupling_min: float = 3.0,
    fix_phi_overcoupled: bool = False,
) -> dict:
    """Fit every trace of a device, then refit traces whose Qc drifted.

    1. Free fit: ``fit_s21_v2`` on every trace (``fit_method = "circle"``); if
       ``Qi <= 0`` and ``negative_qi == "magnitude"``, ``fit_magnitude_fallback``
       (``fit_method = "magnitude"``).
    2. Reference: ``clean_reference(traces, clean_power_min)``.
    3. Drift check on every trace: ``|phi - phi_ref| > phi_tol`` (rad),
       ``|absQc - absQc_ref| / absQc_ref > absqc_rtol``, or ``Qi <= 0``.
    4. If ``auto_fix`` and the trace is overcoupled (``Qi_free / absQc_ref >=
       overcoupling_min``, or ``Qi <= 0``), ``refit_fixed`` with only the
       drifted parameters fixed to ``(ref, sigma)``; a non-positive ``Qi`` fixes
       both. With ``fix_phi_overcoupled``, ``phi`` is also fixed on every other
       overcoupled trace that is not a reference trace, drifted or not.
       ``refit_fixed`` rejects refits that fit much worse than the free fit.

    With fewer than ``min_clean`` clean traces it warns and keeps the free fits.

    Args:
        device: a ``Device``, or a list of one device's traces.
        plot, xpts, phase_err_threshold: passed to ``fit_s21_v2``.
        clean_power_min: minimum power (dBm) of a clean trace.
        phi_tol: allowed ``|phi - phi_ref|`` in rad.
        absqc_rtol: allowed relative deviation of ``absQc`` from ``absQc_ref``.
        min_clean: minimum number of clean traces needed to refit anything.
        auto_fix: if False, only flag drift; never refit.
        refit_background, background_rtol, redchi_ratio_max: passed to
            ``refit_fixed``.
        overcoupling_min: minimum ``Qi_free / absQc_ref`` for a refit. Below
            it (undercoupled or weakly overcoupled), ``Qi`` hardly depends on
            ``phi`` and ``absQc``, so drift is flagged but not refit.
        fix_phi_overcoupled: refit every overcoupled non-reference trace with
            ``phi`` fixed, so ``Qi_err`` carries the ``phi`` systematic even
            when the drift in ``phi`` is too small to detect.
        negative_qi: what to do with a free circle fit with ``Qi <= 0``.
            ``"magnitude"`` runs the magnitude fallback in pass 1.
            ``"circle_fixed"`` keeps the circle fit, so the trace is flagged
            ``"Qi<=0"`` and refit with both ``phi`` and ``absQc`` fixed; the
            magnitude fallback is used only if that refit fails, ``auto_fix``
            is off, or there is no reference.

    Sets on every trace, in addition to the usual fit attributes:
    ``fit_method``; the free-fit values ``Qi_free``, ``Qi_err_free``,
    ``phi_free``, ``phi_err_free``, ``absQc_free``, ``absQc_err_free``,
    ``Ql_free``, ``redchi_free``, ``fit_method_free``, ``background_amp_free``,
    ``background_phase_free``; ``drift`` (comma-separated
    subset of ``"phi"``, ``"absQc"``, ``"Qi<=0"``, empty if none);
    ``refit_rejected`` (``""``, or why a wanted refit was not applied:
    ``"undercoupled"`` (not attempted), ``"redchi"``, ``"Qi<=0"`` or
    ``"error"``); ``background_at_bound`` (None unless a circle refit floated
    the background); and, when a reference exists, ``phi_ref`` and
    ``absQc_ref``. For a ``Device`` it also
    sets ``phi_ref``, ``phi_ref_sigma``, ``absQc_ref``, ``absQc_ref_sigma`` and
    ``n_clean`` on the device (None without a usable reference).

    Returns:
        The reference dict (or None) plus ``"refit"``: ids of refitted traces,
        ``"drifted"``: ids of traces flagged as drifted, and ``"rejected"``:
        ``{id: refit_rejected}`` for wanted refits that were not applied.
    """
    if negative_qi not in ("magnitude", "circle_fixed"):
        raise ValueError(
            f"negative_qi must be 'magnitude' or 'circle_fixed', got {negative_qi!r}"
        )
    is_device = isinstance(device, Device)
    traces = device.traces if is_device else list(device)
    name = device.name if is_device else "traces"
    deferred = set()  # ids of negative-Qi circle fits whose magnitude fallback is deferred

    def run_deferred_fallback(trace):
        if id(trace) in deferred:
            deferred.discard(id(trace))
            fit_magnitude_fallback(trace)

    for trace in traces:
        print(
            f"Fitting trace#{trace.id}, power = {trace.power} dBm, "
            f"temperature = {_temp_mk(trace): .3f} mK"
        )
        fit_s21_v2(trace, plot=plot, xpts=xpts, phase_err_threshold=phase_err_threshold)
        trace.fit_method = "circle"
        if not trace.Qi > 0:
            if negative_qi == "magnitude":
                fit_magnitude_fallback(trace)
            else:
                deferred.add(id(trace))
        for attr in _FREE_ATTRS:
            setattr(trace, f"{attr}_free", getattr(trace, attr))
        trace.drift = trace.refit_rejected = ""
        trace.background_at_bound = None

    ref = clean_reference(traces, clean_power_min)
    if ref is not None and ref["n_clean"] < min_clean:
        ref = None
    if is_device:
        _store(
            device,
            phi_ref=None if ref is None else ref["phi_ref"],
            phi_ref_sigma=None if ref is None else ref["phi_sigma"],
            absQc_ref=None if ref is None else ref["absQc_ref"],
            absQc_ref_sigma=None if ref is None else ref["absQc_sigma"],
            n_clean=0 if ref is None else ref["n_clean"],
        )
    summary = dict(ref or {}, refit=[], drifted=[], rejected={})
    if ref is None:
        n_clean = sum(_is_clean(tr, clean_power_min) for tr in traces)
        warnings.warn(
            f"{name}: only {n_clean} clean traces (min_clean={min_clean}); "
            "keeping the free fits without drift checks",
            RuntimeWarning,
            stacklevel=2,
        )
        for trace in traces:
            run_deferred_fallback(trace)
        return summary

    clean = {id(tr) for tr in traces if _is_clean(tr, clean_power_min)}
    counts = {"phi": 0, "absQc": 0, "Qi<=0": 0}
    for trace in traces:
        trace.phi_ref, trace.absQc_ref = ref["phi_ref"], ref["absQc_ref"]
        drift = _drift(trace, ref, phi_tol, absqc_rtol)
        trace.drift = ",".join(drift)
        coupling = trace.Qi / ref["absQc_ref"]
        overcoupled = not trace.Qi > 0 or coupling >= overcoupling_min
        force_phi = fix_phi_overcoupled and overcoupled and id(trace) not in clean
        if not drift and not force_phi:
            continue
        if drift:
            summary["drifted"].append(trace.id)
        for key in drift:
            counts[key] += 1
        details = []
        if "phi" in drift:
            details.append(f"phi {trace.phi:+.3f} vs {ref['phi_ref']:+.3f}")
        if "absQc" in drift:
            details.append(f"|Qc| {trace.absQc / ref['absQc_ref'] - 1:+.0%}")
        if "Qi<=0" in drift:
            details.append("Qi<=0")
        if force_phi:
            details.append("fix_phi_overcoupled")
        line = (
            f"  trace#{trace.id} {trace.power} dBm {_temp_mk(trace):.1f} mK: "
            f"drift [{', '.join(details)}]"
        )
        if not auto_fix:
            print(line + " (auto_fix=False, not refit)")
            run_deferred_fallback(trace)
            continue
        if not overcoupled:
            trace.refit_rejected = summary["rejected"][trace.id] = "undercoupled"
            print(
                f"{line} -> not refit: Qi/|Qc|_ref = {coupling:.2g} < "
                f"overcoupling_min = {overcoupling_min:g}"
            )
            continue
        fix_phi = "phi" in drift or "Qi<=0" in drift or force_phi
        fix_absqc = "absQc" in drift or "Qi<=0" in drift
        qi_free, redchi_free = trace.Qi, trace.redchi
        ok = refit_fixed(
            trace,
            phi=(ref["phi_ref"], ref["phi_sigma"]) if fix_phi else None,
            absQc=(ref["absQc_ref"], ref["absQc_sigma"]) if fix_absqc else None,
            refit_background=refit_background,
            background_rtol=background_rtol,
            redchi_ratio_max=redchi_ratio_max,
        )
        bound = " (background at bound)" if trace.background_at_bound else ""
        if ok:
            summary["refit"].append(trace.id)
            print(
                f"{line} -> {trace.fit_method}: Qi {qi_free:.3g} -> {trace.Qi:.3g} "
                f"+/- {trace.Qi_err:.2g}, redchi {redchi_free:.2e} -> {trace.redchi:.2e}{bound}"
            )
        else:
            summary["rejected"][trace.id] = trace.refit_rejected
            print(f"{line} -> refit rejected ({trace.refit_rejected}){bound}, free fit kept")
            run_deferred_fallback(trace)

    print(
        f"{name}: phi_ref = {ref['phi_ref']:.3f} +/- {ref['phi_sigma']:.3f} rad, "
        f"|Qc|_ref = {ref['absQc_ref']:.4g} +/- {ref['absQc_sigma'] / ref['absQc_ref']:.1%} "
        f"from {ref['n_clean']} clean traces; drifted {len(summary['drifted'])}/{len(traces)} "
        f"(phi {counts['phi']}, |Qc| {counts['absQc']}, Qi<=0 {counts['Qi<=0']}), "
        f"refit {len(summary['refit'])}, not applied {len(summary['rejected'])}"
    )
    return summary
