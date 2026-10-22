"""Deterministic synthetic devices for the QTLS0 / waterfall / loss-space tests.

The devices mimic strongly overcoupled resonators: the true Qi is ~20x |Qc|, and
each trace's fitted ``phi`` is the true ``phi`` plus a random systematic offset.
``Qi`` is then recomputed from ``Ql`` and the shifted ``phi`` exactly as the circle
fit does, so the phi scatter propagates into ``1/Qi`` as
``d(1/Qi) = sin(phi) / |Qc| * dphi`` while the quoted ``Qi_err`` stays small.
"""

import numpy as np
from lmfit import Parameters

from rrfit.dataio import Device, Trace
from rrfit.fitfns import Qivsnbar, dBmtoW, nbarvsPin
from rrfit.waterfall import _solve_consistent_qint

FR = 5e9
QC = 1e6
PHI0 = 0.3
SIGMA_PHI = 0.013
LINE_ATTENUATION = 70.0

QTLS0_TRUE = dict(qtls0=3e7, nc=1.0, beta=0.8, Qother=1e8)

WATERFALL_TRUE = dict(
    delta_QP0=2e-4, Q_TLS0=3e7, tc=1.2, Q_other=1e8, beta=1.0, beta2=1.0, D_0=100.0
)


def waterfall_params(values=None, **overrides):
    """Waterfall ``Parameters`` with the default bounds and the given values."""
    values = dict(WATERFALL_TRUE if values is None else values, **overrides)
    params = Parameters()
    params.add("delta_QP0", value=values["delta_QP0"], min=0)
    params.add("Q_TLS0", value=values["Q_TLS0"], min=0)
    params.add("tc", value=values["tc"], min=0.0, max=4.5)
    params.add("Q_other", value=values["Q_other"], min=0)
    params.add("beta", value=values["beta"], min=0, max=2.0)
    params.add("beta2", value=values["beta2"], min=0, max=2.0)
    params.add("D_0", value=values["D_0"], min=0)
    return params


def _measured_trace(rng, power, temp, qi_true, qi_err_frac, sigma_phi):
    """Trace whose ``phi`` (and hence ``Qi``) carries a random systematic offset."""
    ql = 1 / (1 / qi_true + np.cos(PHI0) / QC)
    phi = PHI0 + rng.normal(0.0, sigma_phi)
    qi = 1 / (1 / ql - np.cos(phi) / QC)
    return Trace(
        power=power, temperature=temp, fr=FR, Qi=qi, Qi_err=qi_err_frac * qi,
        Ql=ql, absQc=QC, phi=phi, phi_err=1e-4, is_excluded=False,
    )


def overcoupled_qtls0_device(seed=0, qi_err_frac=0.01, sigma_phi=SIGMA_PHI):
    """Base-temperature power sweep generated from ``Qivsnbar``."""
    rng = np.random.default_rng(seed)
    temp = 0.01
    traces = []
    for power in np.linspace(-110.0, -30.0, 17):
        qi = 1e7
        for _ in range(200):
            ql = 1 / (1 / qi + 1 / QC)
            nbar = nbarvsPin(dBmtoW(power - LINE_ATTENUATION), FR, ql, QC)
            qi = Qivsnbar(nbar, *QTLS0_TRUE.values(), FR, temp)
        traces.append(_measured_trace(rng, power, temp, qi, qi_err_frac, sigma_phi))
    return Device(name="synthetic", traces=traces, line_attenuation=LINE_ATTENUATION)


def overcoupled_waterfall_device(seed=0, qi_err_frac=0.01, sigma_phi=SIGMA_PHI):
    """Power x temperature grid generated from the self-consistent waterfall model."""
    rng = np.random.default_rng(seed)
    params = waterfall_params()
    traces = []
    for power in (-90.0, -70.0, -50.0, -30.0):
        for temp in (0.01, 0.05, 0.1, 0.15, 0.2, 0.25):
            qi = _solve_consistent_qint(
                params, temp, FR, dBmtoW(power - LINE_ATTENUATION), QC, 1e7
            )
            traces.append(_measured_trace(rng, power, temp, qi, qi_err_frac, sigma_phi))
    return Device(name="synthetic", traces=traces, line_attenuation=LINE_ATTENUATION)
