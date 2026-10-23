"""Loss-space uncertainties with a phi systematic, for overcoupled resonators.

For a strongly overcoupled resonator, ``Qi = 1 / (1/Ql - cos(phi)/|Qc|)`` is a
small difference of two nearly equal losses. The per-trace ``Qi_err`` from the
fit covariance then underestimates the real uncertainty, ``Qi`` is skewed (it is
~1/x near x = 0), and ``phi`` absorbs normalization systematics that shift the
internal loss by ``d(1/Qi) = sin(phi) / |Qc| * dphi``.

``phi`` is set by the circuit environment and should not depend on power or
temperature, so the scatter of ``phi`` over a device's clean traces estimates
that systematic. ``compute_loss_errors`` turns it into a per-trace loss
uncertainty, which the ``loss_space=True`` options of ``fit_qtls0`` and the
waterfall fits use to fit ``1/Qi`` instead of ``Qi``.

This module only adds attributes to traces and devices; it never changes
``Qi``, ``Qi_err`` or any other existing attribute.
"""

from typing import Callable, Optional

import numpy as np

from rrfit.dataio import Device, Trace

DEFAULT_REFERENCE_MIN_POWER = -60.0


def is_reference_trace(trace: Trace, min_power: float = DEFAULT_REFERENCE_MIN_POWER) -> bool:
    """Default selection of the traces whose ``phi`` scatter defines ``sigma_phi``.

    A trace is a reference if it is not excluded, its power is at least
    ``min_power`` (dBm, as stored on the trace), it is a genuine circle fit, and
    its ``phi`` was fitted rather than fixed by hand. Missing attributes are
    treated permissively: no ``fit_method`` counts as a circle fit, and no
    ``phi_err`` counts as a fitted ``phi``. ``phi_err == 0`` marks a fixed ``phi``.
    """
    if getattr(trace, "is_excluded", False):
        return False
    power = getattr(trace, "power", None)
    if power is None or not power >= min_power:
        return False
    fit_method = getattr(trace, "fit_method", None)
    if fit_method is not None and fit_method != "circle":
        return False
    phi_err = getattr(trace, "phi_err", None)
    if phi_err is not None and phi_err == 0:
        return False
    phi = getattr(trace, "phi", None)
    return phi is not None and bool(np.isfinite(phi))


def estimate_sigma_phi(
    device: Device,
    reference: Optional[Callable[[Trace], bool]] = None,
    min_power: float = DEFAULT_REFERENCE_MIN_POWER,
) -> float:
    """Sample standard deviation (``ddof=1``) of ``phi`` over the reference traces.

    Args:
        device: device whose traces are inspected.
        reference: predicate selecting the reference traces. Defaults to
            ``is_reference_trace(trace, min_power)``.
        min_power: minimum trace power (dBm) for the default predicate.

    Raises:
        ValueError: if fewer than two traces are selected.
    """
    if reference is None:
        def reference(trace):
            return is_reference_trace(trace, min_power)

    phis = np.array([tr.phi for tr in device.traces if reference(tr)], dtype=float)
    if phis.size < 2:
        raise ValueError(
            f"estimate_sigma_phi needs at least 2 reference traces, found {phis.size} "
            f"on device {device.name!r}; relax min_power or pass a reference predicate"
        )
    return float(np.std(phis, ddof=1))


def compute_loss_errors(
    device: Device,
    sigma_phi: Optional[float] = None,
    reference: Optional[Callable[[Trace], bool]] = None,
    min_power: float = DEFAULT_REFERENCE_MIN_POWER,
) -> float:
    """Attach loss-space values and uncertainties to every trace of ``device``.

    Sets ``device.sigma_phi`` and, on each trace, the new attributes

    - ``loss = 1 / Qi``
    - ``loss_err_stat = Qi_err / Qi**2``
    - ``loss_err_sys = |sin(phi)| / absQc * sigma_phi``
    - ``loss_err = sqrt(loss_err_stat**2 + loss_err_sys**2)``

    Traces missing any of ``Qi``, ``Qi_err``, ``phi`` or ``absQc`` get NaN.
    ``Qi`` and ``Qi_err`` are left untouched.

    Args:
        device: device to annotate.
        sigma_phi: phi systematic in radians. If None, it is estimated with
            ``estimate_sigma_phi(device, reference, min_power)``.
        reference, min_power: passed to ``estimate_sigma_phi``.

    Returns:
        The ``sigma_phi`` used.
    """
    if sigma_phi is None:
        sigma_phi = estimate_sigma_phi(device, reference=reference, min_power=min_power)
    device.sigma_phi = float(sigma_phi)

    for trace in device.traces:
        values = [getattr(trace, name, None) for name in ("Qi", "Qi_err", "phi", "absQc")]
        if any(value is None for value in values):
            values = [np.nan] * 4
        qi, qi_err, phi, abs_qc = (float(value) for value in values)
        trace.loss = 1 / qi
        trace.loss_err_stat = qi_err / qi**2
        trace.loss_err_sys = abs(np.sin(phi)) / abs_qc * device.sigma_phi
        trace.loss_err = float(np.hypot(trace.loss_err_stat, trace.loss_err_sys))
    return device.sigma_phi


def loss_arrays(traces) -> tuple[np.ndarray, np.ndarray]:
    """``(loss, loss_err)`` arrays for ``traces``, validated for weighted fitting.

    Raises:
        ValueError: if ``compute_loss_errors`` has not been run, or any trace has
            a non-finite ``1/Qi`` or a non-finite or non-positive ``loss_err``.
    """
    if not all(hasattr(tr, "loss_err") for tr in traces):
        raise ValueError(
            "loss_space=True needs trace.loss_err; call "
            "rrfit.losserr.compute_loss_errors(device) first"
        )
    loss = np.array([tr.loss for tr in traces], dtype=float)
    loss_err = np.array([tr.loss_err for tr in traces], dtype=float)
    if not np.all(np.isfinite(loss) & np.isfinite(loss_err) & (loss_err > 0)):
        raise ValueError(
            "loss_space=True requires finite 1/Qi and finite, positive loss_err "
            "for every non-excluded trace"
        )
    return loss, loss_err


def qi_errorbars_from_loss(loss, loss_err, unbounded_factor=10.0) -> np.ndarray:
    """Asymmetric ``Qi`` error bars, shape ``(2, N)``, from a symmetric loss interval.

    The interval ``loss +/- loss_err`` maps to ``Qi`` in
    ``[1/(loss + loss_err), 1/(loss - loss_err)]``. When ``loss - loss_err <= 0``
    the upper bound is unbounded, and the upper bar is drawn up to
    ``unbounded_factor * Qi`` instead.
    """
    loss = np.asarray(loss, dtype=float)
    loss_err = np.asarray(loss_err, dtype=float)
    qi = 1 / loss
    lower = qi - 1 / (loss + loss_err)
    lo_loss = loss - loss_err
    with np.errstate(divide="ignore"):
        upper = np.where(lo_loss > 0, 1 / lo_loss - qi, (unbounded_factor - 1) * qi)
    return np.vstack([lower, upper])
