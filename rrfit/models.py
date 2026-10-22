""" """

from lmfit import Model
import numpy as np

from rrfit.fitfns import (
    asymmetric_lorentzian,
    cable_delay_linear,
    centered_phase,
    rr_s21_hanger,
    Qivsnbar,
    lossvsnbar,
)


class FitModel(Model):
    """thin wrapper around lmfit's Model class to simplify guessing and fitting"""

    def __init__(self, fitfn, *args, **kwargs):
        """ """
        name = self.__class__.__name__
        super().__init__(func=fitfn, name=name, *args, **kwargs)

    def fit(self, data, x, params=None, verbose=False, **kwargs):
        """ """
        if params is None:
            params = self.guess(data, x)
        result = super().fit(data, params=params, x=x, **kwargs)
        if verbose:
            print(result.fit_report())
        return result

    def make_params(self, guesses: dict = None, **kwargs):
        """ """
        if guesses is not None:
            for param, hint in guesses.items():
                self.set_param_hint(param, **hint)
        return super().make_params(**kwargs)


class S21Model(FitModel):
    """
    s21 fitting function (no background)
    TODO write guess fn
    """

    def __init__(self, *args, **kwargs):
        """ """
        fitfn = rr_s21_hanger
        super().__init__(fitfn, *args, **kwargs)

    def post_fit(self, result):
        """Calculate Ql, absQc, and Qi from fit result best values and show plot"""
        result.params.add("Qi", expr="1 / ((1 / Ql) - (cos(phi) / absQc))")


class S21PhaseLinearModel(FitModel):
    """Fit unwrapped s21 phase to a line to extract cable delay"""

    def __init__(self, *args, **kwargs):
        """ """
        fitfn = cable_delay_linear
        super().__init__(fitfn, *args, **kwargs)

    def guess(self, data, f):
        """ """
        tau_guess = ((data[-1] - data[0]) / (f[-1] - f[0])) / (2 * np.pi)
        theta_guess = np.average(data - 2 * np.pi * f * tau_guess)
        guesses = {
            "theta": {"value": theta_guess},
            "tau": {"value": tau_guess},
        }
        return self.make_params(guesses=guesses)


class S21LogMagModel(FitModel):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        fitfn = asymmetric_lorentzian
        super().__init__(fitfn, *args, **kwargs)

    def guess(self, data, f):
        """ """
        # guess resonance frequency to lie at the center frequency of the acquired sweep
        fr_i = len(f) // 2
        fr_guess = f[fr_i]

        # guess y-offset based on extreme left and right off-resonant data
        xp = len(f) // 6
        l_or, r_or = np.average(data[:xp]), np.average(data[-xp:])
        ofs_guess = (l_or + r_or) / 2

        # guess lorentzian height to be the data range in linear scale
        max_, min_ = np.max(data), np.min(data)
        height_guess = 10 ** ((max_ - ofs_guess) / 20) - 10 ** ((min_ - ofs_guess) / 20)

        # guess fwhm as the frequency difference between the steepest data points
        data_lin = 10 ** (data / 20)
        dl, dr = np.diff(data_lin[:fr_i]), np.diff(data_lin[fr_i:])
        lw_i, rw_i = np.argmax(np.abs(dl)), fr_i + np.argmax(np.abs(dr))
        fwhm_guess = np.abs(f[rw_i] - f[lw_i])

        # guess phi as pi * the ratio of the prominence and range, with a sign
        l_mag, r_mag = data[:fr_i], data[fr_i:]
        prominence, range_ = max_ - ofs_guess, max_ - min_
        sign = 1 if np.average(l_mag) > np.average(r_mag) else -1
        phi_guess = sign * np.pi * prominence / range_

        # set bounds on the guesses and set them as the model's parameter hints
        guesses = {
            "ofs": {"value": ofs_guess},
            "height": {"value": height_guess, "min": 0},
            "phi": {"value": phi_guess, "min": -np.pi, "max": np.pi},
            "fr": {"value": fr_guess, "min": f[0], "max": f[-1]},
            "fwhm": {"value": fwhm_guess, "min": f[1] - f[0], "max": f[-1] - f[0]},
        }
        return self.make_params(guesses=guesses)

    def post_fit(self, result):
        """Calculate Ql, absQc, and Qi from fit result best values and show plot"""
        result.params.add("Ql", expr="fr / fwhm")
        result.params.add("absQc", expr="abs(Ql / height)")
        result.params.add("Qi", expr="1 / ((1 / Ql) - (cos(phi) / absQc))")


class S21CenteredPhaseModel(FitModel):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        fitfn = centered_phase
        super().__init__(fitfn, *args, **kwargs)

    def center_phase(self, centered_s21, discont=1.5 * np.pi, windowx=0.25):
        """transform centered complex S21 into continuous unwrapped centered phase"""
        cphase = np.angle(centered_s21)

        points = len(centered_s21)

        middle = np.argmax(np.abs(np.diff(cphase)))
        left = max(0, middle - int(points * windowx))
        right = min(middle + int(points * windowx), points)

        cphase_unwrapped = np.unwrap(cphase[left:right], discont=discont)
        unwrap_diff_window = cphase_unwrapped - cphase[left:right]
        left_pad = np.repeat(unwrap_diff_window[0], left)
        right_pad = np.repeat(unwrap_diff_window[-1], points - right)

        unwrap_diff = np.concatenate((left_pad, unwrap_diff_window, right_pad))
        cphase += unwrap_diff
        return cphase

    def guess(self, data, f):
        """ """
        # guess resonance frequency to lie at the peak of the derivative of the data
        absdy = np.abs(np.diff(data, prepend=data[0]))
        fr_i = absdy.argmax()
        fr_guess = f[fr_i]

        # guess theta offset based on extreme left and right off-resonant data
        xp = len(f) // 6
        l_or, r_or = np.average(data[:xp]), np.average(data[-xp:])
        theta_guess = (l_or + r_or) / 2

        # guess Ql based on the linewidth of the derivative of the data
        hamp = absdy.max() / 2
        l, r = absdy[:fr_i], absdy[fr_i:]
        lhamp_i = np.abs((l - hamp)).argmin()
        rhamp_i = fr_i + np.abs((r - hamp)).argmin()
        fwhm_guess = f[rhamp_i] - f[lhamp_i]
        Ql_guess = fr_guess / fwhm_guess
        
        sign = -1 if data[0] < data[-1] else 1
        #sign = 1

        fspan = f[-1] - f[0]
        # this fstep estimation works for both linear and homophasal frequency sweeps
        fstep = np.abs(np.average(np.diff(f)))
        guesses = {
            "theta": {"value": theta_guess},
            "fr": {"value": fr_guess, "min": min(f), "max": max(f)},
            "Ql": {"value": Ql_guess, "min": fr_guess / fspan, "max": fr_guess / fstep},
            "sign": {"value": sign, "vary": False},
        }
        return self.make_params(guesses=guesses)

class QivsnbarModel(FitModel):
    """ """

    def __init__(self, fr, temp, *args, **kwargs):
        """ """
        fitfn = Qivsnbar
        self.fr = fr
        self.temp = temp
        super().__init__(fitfn, *args, **kwargs)

    def guess(self, data, x):
        """ """
        qtls0_guess = 10 ** np.floor(np.log10(np.abs(np.min(data))))
        qother_guess = 10 ** np.ceil(np.log10(np.abs(np.max(data))))
        nc_guess = 10 ** np.ceil(np.log10(np.abs(np.min(x))))
        guesses = {
            "qtls0": {"value": qtls0_guess, "min": 0, "max": qother_guess},
            "nc": {"value": nc_guess, "min": 0, "max": max(x)}, 
            "beta": {"value": 1, "min": 0},
            "Qother": {"value": qother_guess, "min": qtls0_guess},
            "fr": {"value": self.fr, "vary": False},
            "temp": {"value": self.temp, "vary": False},
        }
        return self.make_params(guesses=guesses)


class LossvsnbarModel(QivsnbarModel):
    """Fit internal loss 1/Qi vs nbar; same parameters and guesses as QivsnbarModel"""

    def __init__(self, fr, temp, *args, **kwargs):
        """ """
        self.fr = fr
        self.temp = temp
        FitModel.__init__(self, lossvsnbar, *args, **kwargs)

    def guess(self, data, x):
        """guess from Qi = 1 / data, exactly as QivsnbarModel does"""
        return super().guess(1 / np.asarray(data), x)
