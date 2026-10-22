""" """

import numpy as np
from scipy.constants import h, k


def dBmtoW(pow_dbm):
    """ """
    return np.power(10, (pow_dbm - 30) / 10)


def nbarvsPin(Pin, fr, Ql, avgQc):
    """no impedance factor"""
    return (1 / (h * np.pi * fr**2)) * (Ql**2 / avgQc) * Pin


def QTLS(nbar, qtls0, nc, beta, fr, temp):
    """currently used to fit the base temp power sweep data"""
    tanh_term = np.tanh((h * fr) / (k * temp))
    sqrt_term = np.sqrt(1 + np.power(nbar / nc, beta))
    return qtls0 * sqrt_term / tanh_term


def Qivsnbar(x, qtls0, nc, beta, Qother, fr, temp):
    """x: nbar"""
    qtls = QTLS(x, qtls0, nc, beta, fr, temp)
    return 1 / (1 / qtls + 1 / Qother)


def lossvsnbar(x, qtls0, nc, beta, Qother, fr, temp):
    """x: nbar. Internal loss 1/Qi, the reciprocal of ``Qivsnbar``."""
    return 1 / Qivsnbar(x, qtls0, nc, beta, Qother, fr, temp)


def rr_s21_hanger(x, fr, Ql, absQc, phi=0, a=1, alpha=0, tau=0):
    """
    complex s21 model for resonators in hanger/transmission mode
    x: array of probe frequencies (independent variables)
    fr: resonance frequency
    Ql: loaded (total) quality factor
    absQc: absolute value of the coupling quality factor
    phi: phase due to circuit asymmetry
    a: net amplitude loss/gain due to measurement line
    alpha: net phase shift due to measurement line
    tau: frequency-dependent phase shift due to finite measurement line length
    """
    nports = 2
    background = a * np.exp(-1j * (2 * np.pi * x * tau + alpha))
    resonator = (Ql / absQc * np.exp(1j * phi)) / (1 + 2j * Ql * (x / fr - 1))
    return background * (1 - (2 // nports) * resonator)


def asymmetric_lorentzian(x, ofs, height, phi, fr, fwhm):
    """
    Lorentzian fit fn for resonator magnitude response in logarithmic scale
    x: array of probe frequencies (independent variables)
    ofs: y offset
    height: Lorentzian peak/dip to floor distance
    phi: phase factor to account for peak asymmetry
    fr: resonant frequency (peak/dip location)
    fwhm: full width half maxmimum of peak/dip
    """
    numerator = height * np.exp(1j * phi)
    denominator = 1 + 2j * ((x - fr) / fwhm)
    return 20 * np.log10(np.abs(1 - numerator / denominator)) + ofs


def cable_delay_linear(x, tau, theta):
    """
    linear model which gives an initial guess for correcting cable delay tau
    fit this with the unwrapped phase of complex resonator signal
    x: array of probe frequencies (independent variables)
    tau: cable delay
    theta: arbitrary phase offset
    """
    return 2 * np.pi * x * tau + theta


def centered_phase(x, fr, Ql, theta, sign=1):
    """
    Arctan fit for resonator phase response around complex plane origin
    x: array of probe frequencies (independent variables)
    fr: resonant frequency
    Ql: loaded (total) quality factor
    theta: arbitrary phase y-offset
    sign: whether the phase response is S shaped (-1) or not (+1) - fixed
    note that np.arctan return real values in the interval [-pi/2, pi/2]
    """
    return theta + 2 * np.arctan(2 * Ql * sign * (1 - x / fr))