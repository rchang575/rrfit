""" """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from uncertainties.umath import cos as ucos

from rrfit.circlefit import fit_background, fit_circle
from rrfit.models import S21Model, S21CenteredPhaseModel
from rrfit.magfit import fit_magnitude

# TODO plot residuals, separate fit and plot, show s21raw, s21nodelay, s21canonical
# TODO find a way for caller to provide experiment specific figtext


def fit_s21(
    s21, f, plot=False, figsize=(12, 12), weights=None, method="least_squares", **params
):
    """no cable delay correction"""
    s21raw = np.copy(s21)

    # do optional cable delay correction
    tau = params.get("tau", 0)
    s21 *= np.exp(-1j * 2 * np.pi * f * tau)
    s21nodelay = np.copy(s21)

    # find off-resonant point
    discont = params.get("discont", 1.5 * np.pi)
    windowx = params.get("phase_window", 0.25)
    orp = fit_background(s21, f, discont=discont, windowx=windowx)
    s21 /= orp
    s21canonical = np.copy(s21)

    ## circle and phase fit to extract resonator parameters
    radius, center = fit_circle(s21)

    phase_model = S21CenteredPhaseModel()
    if "phase_window" in params:
        windowx = params["phase_window"]
        centered_phase = phase_model.center_phase(
            s21 - center, discont=discont, windowx=windowx
        )
    else:
        centered_phase = np.unwrap(np.angle(s21 - center))
    phase_result = phase_model.fit(centered_phase, f)
    fr_guess = phase_result.best_values["fr"]
    Ql_guess = phase_result.best_values["Ql"]

    phi_guess = -np.arcsin(center.imag / radius)

    # Qc expression valid for transmission/hanger mode
    # for reflection mode, use 1 in the denominator instead of 2
    Qc_guess = Ql_guess / (2 * radius * np.exp(-1j * phi_guess))
    absQc_guess = np.abs(Qc_guess)

    # final lmfit to find final parameter values and fitting uncertainties
    # assume that the background has been fitted out above
    s21_model = S21Model()
    guesses = {
        "fr": {"value": fr_guess},
        "Ql": {"value": Ql_guess},
        "absQc": {"value": absQc_guess},
        "phi": {"value": phi_guess},
        "a": {"value": 1, "vary": False},
        "alpha": {"value": 0, "vary": False},
        "tau": {"value": 0, "vary": False},
    }
    guesses = s21_model.make_params(**guesses)
    s21_result = s21_model.fit(s21, f, params=guesses, weights=weights, method=method)
    print(s21_result.fit_report())
    fit_failed = False
    # extract fit parameters as ufloats
    try:
        fr = s21_result.uvars["fr"]
        absQc = s21_result.uvars["absQc"]
        Ql = s21_result.uvars["Ql"]
        phi = s21_result.uvars["phi"]
        Qi = 1 / ((1 / Ql) - (ucos(phi) / absQc))
    except AttributeError as err:
        print(f"Fit failed. Details: {err}")
        fit_failed = True

    # extract fit parameters and fitting errors
    if not fit_failed:
        fit_params = {
            "fr": fr.n,
            "fr_err": fr.s,
            "Qi": Qi.n,
            "Qi_err": Qi.s,
            "absQc": absQc.n,
            "absQc_err": absQc.s,
            "Ql": Ql.n,
            "Ql_err": Ql.s,
            "phi": phi.n,
            "phi_err": phi.s,
            "background_amp": abs(orp),
            "background_phase": np.angle(orp),
            "chisqr": s21_result.chisqr,
            "redchi": s21_result.redchi,
        }
        fig_title = f"S21 fit [fr = {fr.n:.2g}, Qi = {Qi.n:.2g},"
        fig_title += f" Ql = {Ql.n:.2g}, |Qc| = {absQc.n:.2g}, phi = {phi.n:.2g}]"
    else:
        fit_params = {
            "fr": None,
            "fr_err": None,
            "Qi": None,
            "Qi_err": None,
            "absQc": None,
            "absQc_err": None,
            "Ql": None,
            "Ql_err": None,
            "phi": None,
            "phi_err": None,
            "background_amp": abs(orp),
            "background_phase": np.angle(orp),
            "chisqr": s21_result.chisqr,
            "redchi": s21_result.redchi,
        }
        fig_title = "S21 data, fit errored out"

    if plot:
        fig = plt.figure(tight_layout=True, figsize=figsize)
        fig.suptitle(fig_title)
        gs = GridSpec(6, 6, figure=fig)

        s21_ax = fig.add_subplot(gs[:3, :3])
        s21_ax.set(xlabel="Re(S21)", ylabel="Im(S21)")
        s21_ax.set(title="Circle fit")
        s21_ax.set_aspect("equal", "datalim")
        s21_ax.locator_params(axis="both", nbins=6)
        s21_ax.grid(visible=True, alpha=0.5)
        s21_ax.scatter(s21.real, s21.imag, s=3, c="k", label="data")
        s21_ax.scatter(s21.real[0], s21.imag[0], s=8, c="g")

        s21_cp_ax = fig.add_subplot(gs[:3, 3:])
        s21_cp_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21c) (rad)")
        s21_cp_ax.set(title="Centered phase fit")
        s21_cp_ax.locator_params(axis="both", nbins=6)
        s21_cp_ax.scatter(f, centered_phase, s=3, c="k", label="data")

        s21_mag_ax = fig.add_subplot(gs[3:, :3])
        s21_mag_ax.set(xlabel="Frequency (Hz)", ylabel="|S21| (dB)")
        s21_mag_ax.set(title="Fitted S21 Magnitude")
        s21_mag_ax.locator_params(axis="both", nbins=6)
        s21_mag = 20 * np.log10(np.abs(s21))
        s21_mag_ax.scatter(f, s21_mag, s=3, c="k", label="data")

        s21_phase_ax = fig.add_subplot(gs[3:, 3:])
        s21_phase_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21) (rad)")
        s21_phase_ax.set(title="Fitted S21 Phase")
        s21_phase_ax.locator_params(axis="both", nbins=6)
        s21_phase = np.unwrap(np.angle(s21))
        s21_phase_ax.scatter(f, s21_phase, s=3, c="k", label="data")

        if not fit_failed:
            s21_best_fit = s21_result.best_fit
            s21_ax.plot(
                s21_best_fit.real, s21_best_fit.imag, ls="--", c="r", label="fit"
            )
            cp_best_fit = phase_result.best_fit
            s21_cp_ax.plot(f, cp_best_fit, ls="--", c="r", label="fit")
            s21_mag_fit = 20 * np.log10(np.abs(s21_best_fit))
            s21_mag_ax.plot(f, s21_mag_fit, ls="--", c="r", label="fit")
            s21_phase_fit = np.unwrap(np.angle(s21_best_fit))
            s21_phase_ax.plot(f, s21_phase_fit, ls="--", c="r", label="fit")

        s21_ax.legend()
        s21_cp_ax.legend()
        s21_mag_ax.legend()
        s21_phase_ax.legend()
        plt.show()

    return fit_params


def fit_s21_v2(trace, plot=False, figsize=(12, 12), xpts=5, phase_err_threshold=0.01, plot_title=""):
    s21 = trace.s21real + 1j * trace.s21imag
    s21 *= np.exp(-1j * 2 * np.pi * trace.frequency * trace.tau)
    radius_1, center_1 = fit_circle(s21)

    s21_mag = np.log10(np.abs(s21))
    magfit_result = fit_magnitude(s21_mag, trace.frequency, plot=plot)
    fr_guess = magfit_result.params["fr"].value
    phi_guess = magfit_result.params["phi"].value
    Ql_guess = magfit_result.params["Ql"].value
    # print(magfit_result.fit_report())

    s21c = s21 - center_1
    s21_cp = np.angle(s21c)

    s21cpmodel = S21CenteredPhaseModel()

    # try the naive unwrap
    s21_cp_unwrapped = np.unwrap(np.angle(s21c))
    cp_result_1 = s21cpmodel.fit(
        s21_cp_unwrapped, trace.frequency, method="least_squares"
    )

    # extract relative error of theta
    theta_1 = cp_result_1.params["theta"].value
    theta_1_err = cp_result_1.params["theta"].stderr
    if theta_1_err is None:
        theta_1_relative_err = np.inf
    else:
        theta_1_relative_err = np.abs(theta_1_err / theta_1)
    phase_fit_idx = slice(None, None)
    theta_orp = np.pi
    if theta_1_relative_err > phase_err_threshold:
        left = np.average(s21_cp[:xpts])
        right = np.average(s21_cp[-xpts:])
        theta_orp = (left + right) / 2

        # don't do anything if the circle is already unwrapped
        if 0 < left < np.pi and -np.pi < right < 0:
            theta_orp = np.pi

        s21_unwrapped = s21c * -(np.exp(-1j * (theta_orp)))
        s21_cp_unwrapped = np.angle(s21_unwrapped)
        phase_fit_idx = np.concatenate(
            (
                (s21_cp_unwrapped[trace.frequency < fr_guess] >= 0),
                (s21_cp_unwrapped[trace.frequency > fr_guess] < 0),
            )
        )
        cp_result_1 = s21cpmodel.fit(
            s21_cp_unwrapped[phase_fit_idx],
            trace.frequency[phase_fit_idx],
            method="least_squares",
        )

    #if plot:
    #    cp_result_1.plot(
    #        datafmt=".",
    #        xlabel="Frequency (MHz)",
    #        ylabel="arg(S21) (rad)",
    #        data_kws={"ms": 2, "c": "k"},
    #        fit_kws={"lw": 1.5, "c": "r"},
    #    )
    #print(cp_result_1.fit_report())

    theta = cp_result_1.best_values["theta"] + theta_orp - np.pi
    beta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    rp = center_1 + radius_1 * np.cos(beta) + 1j * radius_1 * np.sin(beta)
    orp = center_1 + (center_1 - rp)

    s21_2 = s21 / orp
    rp_2 = rp / orp
    orp_2 = 1 + 0 * 1j

    radius_2, center_2 = fit_circle(s21_2)

    if plot:
        fig, s21_ax = plt.subplots()

        s21_ax.set(xlabel="Re(S21)", ylabel="Im(S21)")
        s21_ax.set(title="Circle fit (raw data)")
        s21_ax.set_aspect("equal", "datalim")
        s21_ax.locator_params(axis="both", nbins=6)
        s21_ax.grid(visible=True, alpha=0.5)
        s21_ax.scatter(s21.real, s21.imag, s=3, c="k", label="data")
        s21_ax.scatter(s21.real[0], s21.imag[0], s=8, c="g")
        s21_ax.add_patch(
            plt.Circle(
                (center_1.real, center_1.imag),
                radius_1,
                ec="r",
                ls="--",
                fill=False,
                label="fit",
            )
        )
        s21_ax.scatter(rp.real, rp.imag, s=16, c="m")
        s21_ax.scatter(orp.real, orp.imag, s=16, c="r")

        s21_ax.legend()

    phi_guess = -np.arcsin(center_2.imag / radius_2)
    Qc_guess = Ql_guess / (2 * radius_2 * np.exp(-1j * phi_guess))
    absQc_guess = np.abs(Qc_guess)
    s21_model = S21Model()
    guesses = {
        "fr": {
            "value": fr_guess,
            "min": trace.frequency[0],
            "max": trace.frequency[-1],
        },
        "Ql": {"value": Ql_guess, "min": 0},
        "absQc": {"value": absQc_guess, "min": 0},
        "phi": {"value": phi_guess, "min": -np.pi, "max": np.pi},
        "a": {"value": 1, "vary": False},
        "alpha": {"value": 0, "vary": False},
        "tau": {"value": 0, "vary": False},
    }
    guesses = s21_model.make_params(**guesses)
    s21_result = s21_model.fit(
        s21_2, trace.frequency, params=guesses, method="least_squares"
    )
    print(s21_result.fit_report())

    fr = s21_result.uvars["fr"]
    absQc = s21_result.uvars["absQc"]
    Ql = s21_result.uvars["Ql"]
    phi = s21_result.uvars["phi"]
    Qi = 1 / ((1 / Ql) - (ucos(phi) / absQc))

    fit_params = {
        "fr": fr.n,
        "fr_err": fr.s,
        "Qi": Qi.n,
        "Qi_err": Qi.s,
        "absQc": absQc.n,
        "absQc_err": absQc.s,
        "Ql": Ql.n,
        "Ql_err": Ql.s,
        "phi": phi.n,
        "phi_err": phi.s,
        "background_amp": abs(orp),
        "background_phase": np.angle(orp),
        "chisqr": s21_result.chisqr,
        "redchi": s21_result.redchi,
    }

    for name, value in fit_params.items():
        setattr(trace, name, value)

    if plot:
        freq_diff = (trace.frequency - trace.fr) * 1e-3 # kHz

        fig = plt.figure(tight_layout=True, figsize=figsize)
        fig.suptitle(plot_title)
        gs = GridSpec(6, 6, figure=fig)
        s21_ax = fig.add_subplot(gs[:3, :3])
        s21_ax.set(xlabel=r"Re($S_{21}$)", ylabel=r"Im($S_{21}$)")
        s21_ax.set(title="Circle fit")
        s21_ax.set_aspect("equal", "datalim")
        s21_ax.locator_params(axis="both", nbins=6)
        s21_ax.grid(visible=True, alpha=0.5)
        s21_ax.scatter(s21_2.real, s21_2.imag, s=12, c="k", label="data")
        s21_ax.scatter(s21_2.real[0], s21_2.imag[0], s=16, c="g")
        s21_ax.scatter(rp_2.real, rp_2.imag, s=16, c="m")
        s21_ax.scatter(orp_2.real, orp_2.imag, s=16, c="r")
        s21_cp_ax = fig.add_subplot(gs[:3, 3:])
        s21_cp_ax.set(xlabel="Frequency (kHz)", ylabel=r"arg($S_{21}$) (rad)")
        s21_cp_ax.set(title="Centered phase fit")
        s21_cp_ax.locator_params(axis="both", nbins=6)
        s21_mag_ax = fig.add_subplot(gs[3:, :3])
        s21_mag_ax.set(xlabel="Frequency (kHz)", ylabel=r"$|S_{21}|$ (dB)")
        s21_mag_ax.set(title="Fitted S21 Magnitude")
        s21_mag_ax.locator_params(axis="both", nbins=6)
        s21_mag = 20 * np.log10(np.abs(s21_2))
        s21_mag_ax.scatter(freq_diff, s21_mag, s=12, c="k", label="data")
        s21_phase_ax = fig.add_subplot(gs[3:, 3:])
        s21_phase_ax.set(xlabel="Frequency (kHz)", ylabel=r"arg($S_{21}$) (rad)")
        s21_phase_ax.set(title="Fitted S21 Phase")
        s21_phase_ax.locator_params(axis="both", nbins=6)
        s21_phase = np.unwrap(np.angle(s21_2))
        s21_phase_ax.scatter(freq_diff, s21_phase, s=12, c="k", label="data")
        s21_best_fit = s21_result.best_fit
        s21_ax.plot(s21_best_fit.real, s21_best_fit.imag, ls="--", c="r", label="fit")
        s21_cp_ax.scatter(
            freq_diff[phase_fit_idx],
            s21_cp_unwrapped[phase_fit_idx],
            s=12,
            c="k",
            label="data",
        )
        s21_cp_ax.plot(
            freq_diff[phase_fit_idx],
            cp_result_1.best_fit,
            ls="--",
            c="r",
            label="fit",
        )
        s21_mag_fit = 20 * np.log10(np.abs(s21_best_fit))
        s21_mag_ax.plot(freq_diff, s21_mag_fit, ls="--", c="r", label="fit")
        s21_phase_fit = np.unwrap(np.angle(s21_best_fit))
        s21_phase_ax.plot(freq_diff, s21_phase_fit, ls="--", c="r", label="fit")
        s21_ax.legend()
        s21_cp_ax.legend()
        s21_mag_ax.legend()
        s21_phase_ax.legend()
        plt.show()