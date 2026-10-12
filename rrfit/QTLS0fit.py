""" functions to fit QTLS0 from Qint vs nbar or FFS vs temp data"""

import numpy as np
import lmfit
from scipy.constants import k, hbar
import matplotlib.pyplot as plt

from rrfit.dataio import Device
from rrfit.fitfns import dBmtoW, nbarvsPin, Qivsnbar
from rrfit.models import QivsnbarModel
from rrfit.waterfall import _get_device_line_attenuation


def _params_to_value_dict(params):
    return dict(params.valuesdict())


def _params_to_error_dict(params):
    return {
        name: (np.nan if param.stderr is None else param.stderr)
        for name, param in params.items()
    }


def _store_qtls0_fit_summary(device, result):
    device.qtls0_fit_result = result
    device.qtls0_best_values = _params_to_value_dict(result.params)
    device.qtls0_param_errors = _params_to_error_dict(result.params)


def fit_qtls0(device: Device, weighted: bool = True):
    """Fit Qi vs nbar at base temperature to extract QTLS0.

    Args:
        device: device whose non-excluded traces are fitted.
        weighted: if True (default), weight residuals by ``1 / Qi_err``; every
            non-excluded trace must then have a finite, positive ``Qi_err``.
            Pass False for the historical unweighted fit.
    """
    traces = [tr for tr in device.traces if not tr.is_excluded]
    traces.sort(key=lambda x: x.power)
    line_attenuation = _get_device_line_attenuation(device)

    pow_W = np.array([dBmtoW(tr.power - line_attenuation) for tr in traces])
    fr = np.array([tr.fr for tr in traces])
    fr_avg = np.mean(fr)

    temp = np.array([tr.temperature for tr in traces])
    temp_avg = np.mean(temp)

    Qi = np.array([tr.Qi for tr in traces])
    Qi_err = np.array([tr.Qi_err for tr in traces])
    Ql = np.array([tr.Ql for tr in traces])
    avgQc = np.mean(np.array([tr.absQc for tr in traces]))
    avgphi = np.mean(np.array([tr.phi for tr in traces]))
    
    nbar = nbarvsPin(pow_W, fr_avg, Ql, avgQc)

    fit_kwargs = {}
    if weighted:
        Qi_err = np.asarray(Qi_err, dtype=float)
        if not np.all(np.isfinite(Qi_err) & (Qi_err > 0)):
            raise ValueError(
                "fit_qtls0(weighted=True) requires finite, positive Qi_err for every "
                "non-excluded trace"
            )
        fit_kwargs["weights"] = 1 / Qi_err
    result = QivsnbarModel(fr=fr_avg, temp=temp_avg).fit(Qi, nbar, **fit_kwargs)
    print(result.fit_report())

    device.qtls0 = result.best_values["qtls0"]
    device.qtls0_err = result.params["qtls0"].stderr
    device.best_params = result.params
    _store_qtls0_fit_summary(device, result)

    fig, (res_ax, data_ax) = plt.subplots(2, 1, sharex=True, height_ratios=(1, 4))
    #fig.suptitle(f"Device '{device.name}' (pitch {device.pitch}um) QTLS0 from Qi vs nbar")
    nbar_interp = np.linspace(min(nbar), max(nbar), 101)
    qint_interp = Qivsnbar(nbar_interp,
                           result.best_values["qtls0"],
                           result.best_values["nc"],
                           result.best_values["beta"],
                           result.best_values["Qother"],
                           fr_avg, temp_avg)
    res_ax.scatter(nbar, result.residual, s=8, c="k")
    res_ax.set(xlabel="nbar", ylabel="residuals")
    data_ax.errorbar(nbar, Qi, yerr=Qi_err, fmt="ko", label="data")
    #data_ax.errorbar(nbar, Qi, yerr=0, fmt="ko", label="data")
    data_ax.plot(nbar_interp, qint_interp, c="r", ls="--", label="best fit")
    data_ax.set(xlabel="nbar", ylabel="Qi", yscale="log", xscale="log")
    data_ax.legend()
    fig.tight_layout()
