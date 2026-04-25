from scipy.constants import h, k
from scipy.special import kn
from scipy import optimize
from lmfit import Parameters, minimize, report_fit
from rrfit.dataio import Device
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from rrfit.fitfns import nbarvsPin, dBmtoW
import random
import matplotlib.cm as cm


def _default_waterfall_params():
    params = Parameters()
    params.add('delta_QP0', value=2e-4, min=0)
    params.add('Q_TLS0', value=1e6, min=0)
    params.add('tc', value=2.0, min=0.0, max=4.5)
    params.add('Q_other', value=1e7, min=0)
    params.add('beta', value=1, min=0, max=2.0)
    params.add('beta2', value=1, min=0, max=2.0)
    params.add('D_0', value=100, min=0)
    return params


def _active_traces(device: Device):
    return [tr for tr in device.traces if not tr.is_excluded]


def _get_device_line_attenuation(device: Device):
    line_attenuation = getattr(device, "line_attenuation", None)
    if line_attenuation is not None:
        return line_attenuation

    attenuation = getattr(device, "attenuation", 0)
    if attenuation is None:
        return 0
    return attenuation


def _get_waterfall_arrays(device: Device):
    traces = _active_traces(device)
    line_attenuation = _get_device_line_attenuation(device)

    devPowerArray_W = np.array([dBmtoW(tr.power - line_attenuation) for tr in traces])
    tempArray = np.array([tr.temperature for tr in traces])
    freq0Array = np.array([tr.fr for tr in traces])
    QIntArray = np.array([tr.Qi for tr in traces])
    QIntErrArray = np.array([tr.Qi_err for tr in traces])
    Qc = np.mean(np.array([tr.absQc for tr in traces]))
    Ql = np.array([tr.Ql for tr in traces])

    return {
        "traces": traces,
        "power_watts": devPowerArray_W,
        "temperature": tempArray,
        "freq0": freq0Array,
        "qint": QIntArray,
        "qint_err": QIntErrArray,
        "qc": Qc,
        "ql": Ql,
    }


def _normalize_waterfall_fit_result(result, makePlot):
    if makePlot:
        fit_params, initFig, fittedFig = result
        return fit_params, initFig, fittedFig

    if len(result) == 2:
        fit_params, red_chi2 = result
        return fit_params, red_chi2

    if len(result) == 3:
        fit_params, _initFig, _fittedFig = result
        return fit_params, None

    raise ValueError("Unexpected Fit_QIntVsTemp return shape")


def _params_to_value_dict(params):
    return dict(params.valuesdict())


def _params_to_error_dict(params):
    return {
        name: (np.nan if param.stderr is None else param.stderr)
        for name, param in params.items()
    }


def _store_waterfall_fit_summary(device, params):
    device.waterfall_best_values = _params_to_value_dict(params)
    device.waterfall_param_errors = _params_to_error_dict(params)


def _safe_positive(value, floor=1e-12):
    return max(float(value), floor)


def _nbar_from_qint(power, freq0, Qc, qint):
    qint = _safe_positive(qint)
    ql = 1 / (1 / qint + 1 / Qc)
    return nbarvsPin(power, freq0, ql, Qc)


def _nbar_from_inverse_qint(power, freq0, Qc, inv_qint):
    inv_qint = _safe_positive(inv_qint)
    inv_qc = 1 / Qc
    prefactor = power / (h * np.pi * freq0**2)
    return prefactor / (Qc * (inv_qint + inv_qc) ** 2)


def _consistent_qint_residual(qint, outerParams, temp, freq0, power, Qc, fitQP=True):
    delta_QP0 = outerParams['delta_QP0']
    Q_TLS0 = outerParams['Q_TLS0']
    D = outerParams['D_0']
    tc = outerParams['tc']
    Q_other = outerParams['Q_other']
    beta = outerParams['beta']
    beta2 = outerParams['beta2']

    qint = _safe_positive(qint)
    nbar = _nbar_from_qint(power, freq0, Qc, qint)

    QPQ = QPQFunc(temp, delta_QP0, tc, freq0)
    QTLS = QTLSFunc(nbar, Q_TLS0, beta, beta2, D, freq0, temp)
    if fitQP:
        oneOverQ = 1 / QPQ + 1 / QTLS + 1 / Q_other
    else:
        oneOverQ = 1 / QTLS + 1 / Q_other
    q_model = 1 / oneOverQ
    return q_model - qint


def _consistent_inverse_qint_residual(
    inv_qint,
    outerParams,
    temp,
    freq0,
    power,
    Qc,
    fitQP=True,
):
    delta_QP0 = outerParams['delta_QP0']
    Q_TLS0 = outerParams['Q_TLS0']
    D = outerParams['D_0']
    tc = outerParams['tc']
    Q_other = outerParams['Q_other']
    beta = outerParams['beta']
    beta2 = outerParams['beta2']

    inv_qint = _safe_positive(inv_qint)
    nbar = _nbar_from_inverse_qint(power, freq0, Qc, inv_qint)

    inv_qpq = 1 / QPQFunc(temp, delta_QP0, tc, freq0)
    inv_qtls = 1 / QTLSFunc(nbar, Q_TLS0, beta, beta2, D, freq0, temp)
    inv_qother = 1 / Q_other
    if fitQP:
        model_inv_q = inv_qpq + inv_qtls + inv_qother
    else:
        model_inv_q = inv_qtls + inv_qother
    return model_inv_q - inv_qint


def _solve_consistent_qint(
    outerParams,
    temp,
    freq0,
    power,
    Qc,
    qint_init,
    fitQP=True,
):
    qint_init = _safe_positive(qint_init)

    def residual(qint):
        return _consistent_qint_residual(
            qint, outerParams, temp, freq0, power, Qc, fitQP=fitQP
        )

    # Try a fast derivative-free secant solve first.
    try:
        qint_next = max(qint_init * 1.05, qint_init + 1e-9)
        root_result = optimize.root_scalar(
            residual,
            method="secant",
            x0=qint_init,
            x1=qint_next,
            maxiter=50,
        )
        if root_result.converged and np.isfinite(root_result.root):
            return _safe_positive(root_result.root)
    except (RuntimeError, ValueError, OverflowError, FloatingPointError):
        pass

    # Fall back to minimizing the squared residual over a wide positive interval.
    upper_bound = max(
        qint_init * 100,
        float(outerParams["Q_other"]) * 10,
        float(outerParams["Q_TLS0"]) * 10,
        1.0,
    )
    lower_bound = max(min(qint_init / 100, upper_bound / 1e6), 1e-12)
    min_result = optimize.minimize_scalar(
        lambda qint: residual(qint) ** 2,
        bounds=(lower_bound, upper_bound),
        method="bounded",
        options={"maxiter": 100},
    )
    if not min_result.success or not np.isfinite(min_result.x):
        raise ValueError("consistent Qint solver failed to converge")
    return _safe_positive(min_result.x)


def _solve_consistent_qint_reduced(
    outerParams,
    temp,
    freq0,
    power,
    Qc,
    qint_init,
    fitQP=True,
):
    inv_qint_init = 1 / _safe_positive(qint_init)

    def residual(inv_qint):
        return _consistent_inverse_qint_residual(
            inv_qint,
            outerParams,
            temp,
            freq0,
            power,
            Qc,
            fitQP=fitQP,
        )

    lower = _safe_positive(min(inv_qint_init / 10, 1e-12))
    upper = max(inv_qint_init * 10, lower * 10)
    f_lower = residual(lower)
    f_upper = residual(upper)

    for _ in range(12):
        if np.sign(f_lower) != np.sign(f_upper):
            break
        upper *= 10
        f_upper = residual(upper)
    else:
        return _solve_consistent_qint(
            outerParams,
            temp,
            freq0,
            power,
            Qc,
            qint_init,
            fitQP=fitQP,
        )

    root_result = optimize.root_scalar(
        residual,
        bracket=(lower, upper),
        method="brentq",
        maxiter=100,
    )
    if not root_result.converged or not np.isfinite(root_result.root):
        raise ValueError("reduced consistent Qint solver failed to converge")
    return 1 / _safe_positive(root_result.root)


def _solve_consistent_qint_dispatch(
    outerParams,
    temp,
    freq0,
    power,
    Qc,
    qint_init,
    fitQP=True,
    consistent_method="qint",
):
    if consistent_method == "qint":
        return _solve_consistent_qint(
            outerParams,
            temp,
            freq0,
            power,
            Qc,
            qint_init,
            fitQP=fitQP,
        )
    if consistent_method == "inverse_qint":
        return _solve_consistent_qint_reduced(
            outerParams,
            temp,
            freq0,
            power,
            Qc,
            qint_init,
            fitQP=fitQP,
        )
    raise ValueError(f"Unknown consistent_method: {consistent_method}")


def predict_qint_from_fixed_nbar(temp, params, freq0, nbar, powerID=0):
    return QIntVsTemp_TLS_QP_Beta_fit_usingParams(temp, params, freq0, nbar, powerID)


def _consistent_tail_needs_refinement(qint_values, tail_convergence_tol=1e-3):
    if len(qint_values) < 2:
        return False
    if qint_values[-2] == 0:
        return False
    return abs(qint_values[-1] / qint_values[-2] - 1) < tail_convergence_tol


def predict_qint_consistent_curve(
    temp,
    params,
    freq0,
    power,
    Qc,
    qint_init,
    consistent_method="qint",
    max_refinement_passes=6,
    tail_convergence_tol=1e-3,
):
    prediction = np.asarray(qint_init, dtype=float)
    attempts = max(1, max_refinement_passes)

    for _ in range(attempts):
        prediction = QIntVsTemp_consistent(
            temp,
            params,
            freq0,
            power,
            Qc,
            prediction,
            consistent_method=consistent_method,
        )
        if not _consistent_tail_needs_refinement(prediction, tail_convergence_tol):
            break

    return prediction


def make_local_bounds(boundsDict, params, shrink=0.25):
    local_bounds = {}
    for param, (lower, upper) in boundsDict.items():
        span = upper - lower
        half_window = 0.5 * span * shrink
        center = params[param].value
        local_lower = max(lower, center - half_window)
        local_upper = min(upper, center + half_window)
        if local_lower == local_upper:
            local_lower = lower
            local_upper = upper
        local_bounds[param] = (local_lower, local_upper)
    return local_bounds


def _build_params_from_vector(base_params, param_names, values):
    params = base_params.copy()
    for name, value in zip(param_names, values):
        params[name].value = value
    return params


def _waterfall_global_objective(
    values,
    base_params,
    param_names,
    tempArray,
    freq0Array,
    devPowerArray_W,
    Qc,
    QIntArray,
    QIntErrArray,
    consistent=True,
    consistent_method="qint",
):
    params = _build_params_from_vector(base_params, param_names, values)
    if consistent:
        resid = QIntVsTemp_consistent_error_function(
            params,
            tempArray,
            freq0Array,
            devPowerArray_W,
            Qc,
            QIntArray,
            QIntArray,
            QIntErrArray,
            consistent_method,
        )
    else:
        ql = 1 / (1 / QIntArray + 1 / Qc)
        nbarArray = nbarvsPin(devPowerArray_W, freq0Array, ql, Qc)
        resid = QIntVsTemp_TLS_QP_Beta_error_function_usingParams(
            params,
            tempArray,
            QIntArray,
            freq0Array,
            nbarArray,
            0,
            QIntErrArray,
        )
    resid = np.asarray(resid, dtype=float)
    return float(np.dot(resid, resid))


def fit_waterfall_global(
    device,
    boundsDict,
    init_params=None,
    consistent=True,
    consistent_method="qint",
    makePlot=True,
    maxiter=40,
    popsize=10,
    tol=0.01,
    polish=False,
    seed=None,
):
    """
    Experimental waterfall fit:
    1. optimize the chosen objective globally over the physical parameters
    2. run the existing local Fit_QIntVsTemp refinement from that best point

    This preserves the legacy/staged workflows while providing a more principled
    initializer than repeated uniform random restarts.
    """
    if init_params is None:
        init_params = _default_waterfall_params()

    arrays = _get_waterfall_arrays(device)
    param_names = list(boundsDict.keys())
    bounds = [tuple(boundsDict[name]) for name in param_names]

    de_result = optimize.differential_evolution(
        _waterfall_global_objective,
        bounds=bounds,
        args=(
            init_params.copy(),
            param_names,
            arrays["temperature"],
            arrays["freq0"],
            arrays["power_watts"],
            arrays["qc"],
            arrays["qint"],
            arrays["qint_err"],
            consistent,
            consistent_method,
        ),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=polish,
        seed=seed,
        init="latinhypercube",
    )

    seeded_params = _build_params_from_vector(init_params, param_names, de_result.x)
    device.waterfall_global_result = de_result
    device.best_params = seeded_params.copy()

    if makePlot:
        fit_params, initFig, fittedFig = _normalize_waterfall_fit_result(
            Fit_QIntVsTemp(
                device,
                seeded_params,
                consistent=consistent,
                consistent_method=consistent_method,
                makePlot=True,
            ),
            makePlot=True,
        )
    else:
        fit_params, _ = _normalize_waterfall_fit_result(
            Fit_QIntVsTemp(
                device,
                seeded_params,
                consistent=consistent,
                consistent_method=consistent_method,
                makePlot=False,
            ),
            makePlot=False,
        )
        initFig = None
        fittedFig = None
    device.best_params = fit_params
    return fit_params, initFig, fittedFig


def QTLSFunc(nbar, qtls0, beta_1, beta_2, D, fr, temp):
    tanh_term = np.tanh((h * fr) / (2 * k * temp))
    sqrt_term = np.sqrt(1 + (np.power(nbar, beta_2) / (D * np.power(temp, beta_1))) * tanh_term)
    return qtls0 * sqrt_term / tanh_term

def QPQFunc(tempK, delta_QP0, tc, fr):
    gap = tc * 1.764 * k
    oneOverQPQ = (delta_QP0) * np.exp(-gap/(k*tempK)) * np.sinh(h*fr / (2*k*tempK)) * kn(0, h*fr / (2*k*tempK))
    return 1/oneOverQPQ

# error function to solve the transcendental equation for Q_int
def consistentQintError(params, outerParams, temp, freq0, power, Qc, fitQP = True):
    qint = params['Qint']
    return _consistent_qint_residual(
        qint, outerParams, temp, freq0, power, Qc, fitQP=fitQP
    ) ** 2


def QIntVsTemp_consistent(
    temp,
    params,
    freq0,
    power,
    Qc,
    Qint_init,
    consistent_method="qint",
):
    QInt = np.zeros(np.size(power))
    for i, currentPower in enumerate(power):
        QInt[i] = _solve_consistent_qint_dispatch(
            params,
            temp[i],
            freq0[i],
            currentPower,
            Qc,
            Qint_init[i],
            consistent_method=consistent_method,
        )
    return QInt


def QIntVsTemp_consistent_error_function(
    params,
    temps,
    freq0,
    power,
    Qc,
    Qint_init,
    data,
    errors,
    consistent_method="qint",
):
    resid = []

    for i in range(len(temps)):
        res = (
            data[i]
            - QIntVsTemp_consistent(
                [temps[i]],
                params,
                [freq0[i]],
                [power[i]],
                Qc,
                [Qint_init[i]],
                consistent_method=consistent_method,
            )[0]
        ) / errors[i]
        #res = (data[i] - QIntVsTemp_consistent([temps[i]], params, [freq0[i]], [power[i]], Qc, [Qint_init[i]])[0])
        #print(f"{res}")
        resid.append(res)
    return np.hstack(resid)

# fit function left by Alex. Note that it does not calculate nbar as a function of Qint, but assumes that nbar is a
# fixed number. We were calculating nbar using the Qint measured from the data
def QIntVsTemp_TLS_QP_Beta_fit_usingParams(temp, params, freq0, nbar, powerID, zResonator=None):
    delta_QP0 = params['delta_QP0']
    Q_TLS0    = params['Q_TLS0']
    D         = params['D_%i'%powerID]
    tc       = params['tc']
    Q_other   = params['Q_other']
    beta      = params['beta']
    try:
        beta2     = params['beta2']
    except:
        beta2 = 1.0
    #omega     = 2 * np.pi * freq0
    #temp = tempMK * 1e-3

    QPQ = QPQFunc(temp, delta_QP0, tc, freq0)
    QTLS = QTLSFunc(nbar, Q_TLS0, beta, beta2, D, freq0, temp)
    oneOverQ = 1/QPQ + 1/QTLS + 1/Q_other
    Q = 1/oneOverQ
    
    #if np.isnan(Q):
    #    print([(p.name, p.value) for p in params.values()])
    #    print(freq0)
    #    print(nbar)
    #    print(QPQ)
    #    print(QTLS)
    #    print(temp)
    return Q

def QIntVsTemp_TLS_QP_Beta_error_function_usingParams(params, temps, data, freq0, nbarLis, powerIDs, errors):
    resid = []
    for i in range(len(temps)):
        val = (
            data[i]
            - QIntVsTemp_TLS_QP_Beta_fit_usingParams(
                temps[i], params, freq0[i], nbarLis[i], powerIDs
            )
        ) / errors[i]
        #val = (data[i] - QIntVsTemp_TLS_QP_Beta_fit_usingParams(temps[i], params, freq0[i], nbarLis[i], powerIDs))
        resid.append(val)
    return np.hstack(resid)

def plot_Qi_vs_temp(
    device: Device,
    figsize=(12, 8),
    plotParams=None,
    fitFunc=QIntVsTemp_consistent,
    consistent_method="qint",
):
    """ """
    
    traces = _active_traces(device)
    min_temp = min([tr.temperature for tr in traces])
    max_temp = max([tr.temperature for tr in traces])
    data = defaultdict(list)
    for trace in traces:
        data[trace.power].append(trace)

    fig, ax = plt.subplots(figsize=figsize)    
    #fig.suptitle(f"Device {device.name} (pitch = {device.pitch}um): Qi vs temp")

    uniquePowers = {tr.power for tr in device.traces if not tr.is_excluded}
    C = cm.rainbow(np.linspace(1, 0, len(uniquePowers)))

    for idx, (power, traces) in enumerate(sorted(data.items(), reverse=True)):
        traces.sort(key=lambda x: x.temperature)
        temp = np.array([tr.temperature for tr in traces])
        Qi = np.array([tr.Qi for tr in traces])
        Qi_err = np.array([tr.Qi_err for tr in traces])
        Qc = np.mean(np.array([tr.absQc for tr in traces]))
        Ql = np.array([tr.Ql for tr in traces])
        freq0List = np.array([tr.fr for tr in traces])
        line_attenuation = _get_device_line_attenuation(device)
        devPowerArray_W = np.array([dBmtoW(tr.power - line_attenuation) for tr in traces])

       #ax.errorbar(temp, Qi, yerr=Qi_err, mec=f"C{idx}", ls="", mfc=f"C{idx}", marker="o", ms=6, label=f"{power:.1f} dBm")

        color = C[idx] #f"C{idx}"

        if not plotParams is None:
            if fitFunc == QIntVsTemp_consistent:
                tempAxis = np.linspace(min_temp, max_temp, 100)
                freq0Interp = np.interp(tempAxis, temp, freq0List)
                QIntInterp = np.interp(tempAxis, temp, Qi)
                ys = predict_qint_consistent_curve(
                    tempAxis,
                    plotParams,
                    freq0Interp,
                    np.ones(np.size(tempAxis)) * devPowerArray_W[0],
                    Qc,
                    QIntInterp,
                    consistent_method=consistent_method,
                )

                # plot the final data
                ax.plot(tempAxis * 1e3, ys, label='{0:.1f} dBm'.format(power),
                        color=color, linewidth=2)
            else:
                # calculate the nbar array from data
                nbarArray = nbarvsPin(devPowerArray_W, freq0List, Ql, Qc)
                ys = predict_qint_from_fixed_nbar(temp, plotParams, freq0List, nbarArray, 0)
                ax.plot(temp * 1e3, ys, label='{0:.1f} dBm'.format(power),
                        color=color, linewidth=2)
            
            ax.errorbar(temp * 1e3, Qi, Qi_err,
                        marker='o', markersize=8, alpha=0.8,
                        linewidth=0,
                        markeredgecolor=color, markeredgewidth=1,
                        markerfacecolor=color, zorder=1, elinewidth=1,
                        capsize=4, ecolor='k')

        else:
            ax.errorbar(temp * 1e3, Qi, Qi_err,
                        marker='o', markersize=8, alpha=0.8,
                        linewidth=0,
                        markeredgecolor=color, markeredgewidth=1,
                        markerfacecolor=color, zorder=1, elinewidth=1,
                        capsize=4, ecolor='k',
                        label='{0:.1f} dBm'.format(power))

    ax.tick_params(axis='both', which='major', labelsize=30, size=8, width=2)
    ax.tick_params(which='minor', size=4, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel(r"Temperature (mK)", fontsize=30)
    ax.set_ylabel(r"$Q_{int}$", fontsize=30)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    #plt.show()
    return fig, ax

####################################################################################################################
# Functions written by Russell McLellan, Sept 11, 2022
# fitIterated will try to repeatedly fit a dataset with random starting values. The best fit is stored in the self
# object for later use. The purpose of the function is to be more robust to local minima - rather than rely on a
# hand picked starting value, we randomly seed the parameter space and find the lowest spot
# createFitHistograms will display the results of fitIterated
# these two functions were tested for the Q vs T data, but I believe the structure is general enough to use with any
# fit function if we turn the 'Fit_QIntVsTemp' function call to different functions and adjust createFitHistograms
# to properly display different numbers of parameters in the histogram arrays
# I assume that we use the lmfit module to fit the data.
# inputs:
#   boundsDict - dictionary with the same names as the lmfit Parameters object. boundsDict[<param name>][0] is the
#                lower bound of the parameter range the algorithm will guess, and boundsDict[<param name>][1] is the
#                upper bound
#   numIter - number of guesses you want to try
#   fitFunc, errorFunc - functions for the fit. fitFunc returns the data, errorFunc the residuals
#   makePlot - boolean - set to True to plot the results, False to only store the data
# outputs:
#   initDict - dictionary with same names as the lmfit Parameters object. Each entry is a numIter length numpy array
#              corresponding to an initial guess of the parameter. initDict[<parameter1>][i],
#              initDict[<parameter2>][i], initDict[<parameter3>][i], etc correspond to a single guess of the set of
#              parameters.
#   finalDict - same as initDict, but saves the result of the fit
#   red_chi2 - numpy array, length numIter - reduced chi2 values corresponding to the fits
#   and a list of figures created if makePlot=True
def fitIterated(device, boundsDict, numIter, consistent=False, makePlot=True, fitQP=True, retries = 10, init_params=None):

    if init_params is None:
        init_params = _default_waterfall_params()
    setattr(device, "best_params", init_params.copy())

    # initialize output variables
    initDict = {}
    finalDict = {}
    for param in boundsDict.keys():
        initDict[param] = np.zeros(numIter)
        finalDict[param] = np.zeros(numIter)
    red_chi2_arr = np.zeros(numIter)
    best_fit_params = None
    best_red_chi2 = np.inf
    # run the fits in a loop
    for i in range(numIter):
        print(f"Running iteration {i+1}/{numIter}...")
        # we sometimes encounter a ValueError if the fit is seeded with initial guesses that cause NaN. I set up a
        # while loop with a try/except block to redo any guess that throws a ValueError.
        # Note that there is no breaking from this While - if boundsDict is set too poorly, the program will hang
        # AJ - added a while loop break if exceed `retries``
        check = True
        retry_count = 0
        while check:
            try:
                for param in boundsDict.keys():
                    initDict[param][i] = random.uniform(boundsDict[param][0], boundsDict[param][1])
                    init_params[param].value = initDict[param][i]
                    # AJ - added to try and avoid nan values, need to eventually remove this because it does not make sense to bound params based on initial guess values
                    # param bounds and initial guess range should be independent knobs
                    #init_params[param].min = boundsDict[param][0]
                    #init_params[param].max = boundsDict[param][1]
                #print(f"Outside minimize: {[(p.name, p.value) for p in init_params.values()]}")
                params, red_chi2 = _normalize_waterfall_fit_result(
                    Fit_QIntVsTemp(
                        device,
                        init_params,
                        consistent=consistent,
                        makePlot=False,
                    ),
                    makePlot=False,
                )
                for param in boundsDict.keys():
                    finalDict[param][i] = params[param].value
                red_chi2_arr[i] = red_chi2
                if red_chi2 < best_red_chi2:
                    best_red_chi2 = red_chi2
                    best_fit_params = params.copy()
                check = False
            except ValueError as err:
                retry_count += 1
                if retry_count >= retries:
                    print(f"Fit aborted! Exceeded {retries = } for {err = }")
                    return
            #    pass
    # create summary plots, if requested
    if makePlot:
        chi2Fig, countFig, probFig = createFitHistograms(device, initDict, finalDict, boundsDict, red_chi2_arr)
    else:
        chi2Fig = None
        countFig = None
        probFig = None
    # save the best fit in the hanger object
    if best_fit_params is None:
        raise RuntimeError("fitIterated did not complete a successful fit")
    device.best_params = best_fit_params.copy()
    if makePlot:
        final_fit_params, initFig, fittedFig = _normalize_waterfall_fit_result(
            Fit_QIntVsTemp(
                device,
                device.best_params,
                consistent=consistent,
                makePlot=True,
            ),
            makePlot=True,
        )
    else:
        final_fit_params, _ = _normalize_waterfall_fit_result(
            Fit_QIntVsTemp(
                device,
                device.best_params,
                consistent=consistent,
                makePlot=False,
            ),
            makePlot=False,
        )
        initFig = None
        fittedFig = None
    device.best_params = final_fit_params
    return initDict, finalDict, red_chi2_arr, [chi2Fig, countFig, probFig, initFig, fittedFig]


def fit_waterfall_staged(
    device,
    boundsDict,
    coarse_iterations=30,
    refine_iterations=0,
    local_shrink=0.25,
    makePlot=True,
    retries=10,
    init_params=None,
):
    """
    Faster waterfall workflow:
    1. broad non-consistent random search to find a good basin cheaply
    2. optional narrow consistent search around the coarse best fit
    3. final consistent fit for the trusted result

    This mirrors the final-call pattern of Fit_QIntVsTemp:
    fit_params, initFig, fittedFig = fit_waterfall_staged(...)
    device.best_params = fit_params
    """
    if init_params is None:
        init_params = _default_waterfall_params()

    coarse_result = fitIterated(
        device,
        boundsDict,
        coarse_iterations,
        consistent=False,
        makePlot=False,
        retries=retries,
        init_params=init_params.copy(),
    )
    coarse_best_params = device.best_params.copy()

    refine_result = None
    if refine_iterations > 0:
        refine_bounds = make_local_bounds(boundsDict, coarse_best_params, shrink=local_shrink)
        refine_result = fitIterated(
            device,
            refine_bounds,
            refine_iterations,
            consistent=True,
            makePlot=False,
            retries=retries,
            init_params=coarse_best_params.copy(),
        )

    device.waterfall_staged_result = {
        "coarse_search": coarse_result,
        "consistent_refine": refine_result,
    }

    final_fit_params, initFig, fittedFig = Fit_QIntVsTemp(
        device,
        device.best_params.copy(),
        consistent=True,
        makePlot=makePlot,
    )
    device.best_params = final_fit_params

    return final_fit_params, initFig, fittedFig

# inputs are taken from the fitIterated function except for:
#   probCutoff - probability cutoff for chi2 values. If a fit has a probability lower than probCutoff of being
#                correct (relative to the best fit), then it is excluded from the plot. Without probCutoff you end
#                up plotting some terrible fits that make the visualization break. We only want to look at fits that
#                are reasonably good
def createFitHistograms(device, initDict, finalDict, boundsDict, red_chi2, probCutoff=0.1):
    plt.rcParams['font.size'] = 12
    import scipy.stats as stats
    nfree = device.waterfall_fit_result.nfree
    prob = stats.chi2.pdf(red_chi2 / min(red_chi2) * nfree, nfree) / \
           stats.chi2.pdf(nfree, nfree)  # normalized probability to best result
    indsToKeep = np.where(prob > probCutoff)[0]  # indices where the probability is greater than probCutoff
    # histogram of chi2 values
    chi2Fig, hax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=150)
    hax.hist(red_chi2[indsToKeep], 30, alpha=0.7)
    hax.set_xlabel(r'$\chi^2$', labelpad=20)
    hax.set_ylabel('Counts')
    # histograms of raw counts
    countFig, hax = plt.subplots(2, 4, figsize=(10, 6), dpi=150)
    hax = np.concatenate((hax[0], hax[1]))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, param in enumerate(initDict.keys()):
        myBins = np.linspace(min([boundsDict[param][0], min(finalDict[param][indsToKeep])]),
                             max([boundsDict[param][1], max(finalDict[param][indsToKeep])]),
                             30)
        hax[i].hist(initDict[param][indsToKeep], myBins, label='initial', alpha=0.7)
        hax[i].hist(finalDict[param][indsToKeep], myBins, label='final', alpha=0.7)
        hax[i].axvline(x=boundsDict[param][0], color='k', linestyle='--', label='initial guess bounds', linewidth=2)
        hax[i].axvline(x=boundsDict[param][1], color='k', linestyle='--', linewidth=2)
        hax[i].set_xlabel(param)
        hax[i].set_ylabel('Counts')
        if i == 6:  # add legend to last plot only
            hax[i].legend(bbox_to_anchor=(1, 1))
    hax[-1].set_axis_off()
    plt.suptitle('Counts histograms')
    # histograms of probability weighted data
    # The weighting will suppress any local minima results with a low probability of being correct
    probFig, hax = plt.subplots(2, 4, figsize=(10, 6), dpi=150)
    hax = np.concatenate((hax[0], hax[1]))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, param in enumerate(initDict.keys()):
        myBins = np.linspace(min([boundsDict[param][0], min(finalDict[param][indsToKeep])]),
                             max([boundsDict[param][1], max(finalDict[param][indsToKeep])]),
                             30)
        hax[i].hist(finalDict[param][indsToKeep], myBins, weights=prob[indsToKeep], label='final', alpha=0.7,
                    color='C1')
        hax[i].set_xlabel(param)
        hax[i].legend()
        hax[i].set_ylabel('Counts*prob')
    hax[-1].set_axis_off()
    plt.suptitle('Counts*prob histograms')
    return chi2Fig, countFig, probFig

def Fit_QIntVsTemp(
    device,
    init_params,
    consistent=False,
    consistent_method="qint",
    makePlot=True,
):
    # rewritten from scratch by Russell, Sept 7, 2022
    # there is a separate set of calls to deal with the case when the QInt calculation is done in a self-consistent
    # manner using fitFunc=QIntVsTemp_consistent, errorFunc=QIntVsTemp_consistent_error_function. That's the only
    # way to get a smoothly varying curve.
    # TODO: add analytic way of varying freq0 for the plot instead of interpolations?
    # inputs:
    #   consistent - boolean. Set to False to calculate nbar from data, True to calculate nbar and Q simultaneously
    #   makePlot - boolean that sets whether plots are generated
    # find arrays of things for later
    
    arrays = _get_waterfall_arrays(device)
    tempArray = arrays["temperature"]
    freq0Array = arrays["freq0"]
    devPowerArray_W = arrays["power_watts"]
    QIntArray = arrays["qint"]
    QIntErrArray = arrays["qint_err"]
    Qc = arrays["qc"]
    Ql = arrays["ql"]

    # calculate the nbar array for use in the fit
    nbarArray = nbarvsPin(devPowerArray_W, freq0Array, Ql, Qc)

    #### Fit
    if consistent:
        print("Starting consistent fit...")
        out_main = minimize(QIntVsTemp_consistent_error_function, init_params, \
                args=(
                    tempArray,
                    freq0Array,
                    devPowerArray_W,
                    Qc,
                    QIntArray,
                    QIntArray,
                    QIntErrArray,
                    consistent_method,
                ),
                method="least_squares")
        print("Done consistent fit")
        #if out_main.params['Q_other'].stderr is None:
        #    self.initParams_QIntVsTemp.pop('Q_other')
        #    init_params = self.initParams_QIntVsTemp
        #    out_main = minimize(QIntVsTemp_consistent_error_function_noQother, init_params, \
        #                        args=(
        #                        tempArray, freq0Array, devPowerArray_W, Qc, QIntArray, QIntArray, QIntErrArray,
        #                        self.zResonator))
        #    self.initParams_QIntVsTemp.add('Q_other', value=np.Inf, min=0, vary=False)
        #    out_main.params.add('Q_other', value=np.Inf, min=0, vary=False)
    else:
        out_main = minimize(QIntVsTemp_TLS_QP_Beta_error_function_usingParams, init_params, \
                args=(tempArray, QIntArray, freq0Array, nbarArray, 0, QIntErrArray),
                method="least_squares")
    fit_params = out_main.params
    final_red_chi2 = out_main.redchi
    device.waterfall_fit_result = out_main
    _store_waterfall_fit_summary(device, fit_params)

    #### plot
    if makePlot:
        if consistent:
            fitFunc = QIntVsTemp_consistent
        else:
            fitFunc = QIntVsTemp_TLS_QP_Beta_fit_usingParams
        initFig = None #initFig, ax = plot_Qi_vs_temp(device, fitFunc=fitFunc, plotParams=init_params)
        #ax.set_title('Init params, ' + device.name)
        fittedFig, ax = plot_Qi_vs_temp(
            device,
            fitFunc=fitFunc,
            plotParams=out_main.params,
            consistent_method=consistent_method,
        )
        #ax.set_title('Fitted params, ' + device.name)
        # Print out fit parameters:
        report_fit(out_main)
        print('Reduced Chi Squared: {0:.2f}'.format(out_main.redchi))
        return fit_params, initFig, fittedFig
    else:
        return fit_params, final_red_chi2
