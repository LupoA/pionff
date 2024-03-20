import numpy as np
from scipy.integrate import quad


def _omnes_rep_integrand(sp, s, phaseshift, *args):
    """
    returns delta(s') / s' (s'-s) to be integrated in s'
    args: positional arguments of phaseshift
    """
    res = phaseshift(np.sqrt(sp), *args)
    res /= sp * (sp - s)
    return res


def _phase_shift_reduced_support(t, s_th, phaseshift, *args):
    """
    Phaseshift, after change of variable:
    s = s_th (1+t)/(1-t)
    """
    _s = s_th * (1 + t) / (1 - t)
    return phaseshift(np.sqrt(_s), *args)


def _omnes_rep_integrand_reduced_support(t, s, s_th, phaseshift, *args):
    """
    Integrand for Omnes, after change of variable:
    s = s_th (1+t)/(1-t)
    to be integrated in [0,1] with a pole in t_pole
    """
    _prefactor = 2 / (s + s_th)
    res = _phase_shift_reduced_support(t, s_th, phaseshift, *args) / (1 + t)
    return res * _prefactor


def _t_pole(s, s_th):
    return (s - s_th) / (s + s_th)


def omnes_below_threshold_scalar(s, s_th, phaseshift, *args):
    """
    args: positional arguments of phaseshift
    s_th = 4 m_pi^2
    """
    res, _err = quad(_omnes_rep_integrand, s_th, np.inf, args=(s, phaseshift, *args))
    res *= s / np.pi
    return np.exp(res)


omnes_below_threshold = np.vectorize(omnes_below_threshold_scalar)


def omnes_above_threshold(s, s_th, phaseshift, *args, cut=None):
    """
    args: positional arguments of phaseshift
    s_th = 4 m_pi^2
    principal value
    """
    t_pole = _t_pole(s, s_th)
    res, _err = quad(
        lambda x: _omnes_rep_integrand_reduced_support(x, s, s_th, phaseshift, *args),
        weight="cauchy",
        wvar=t_pole,
        a=0,
        b=1 - 1e-6,
    )

    res *= s / np.pi
    return np.exp(res)


def omnes_function_scalar(s, s_th, phaseshift, *args, cut=None):
    """
    splitting is needed to avoid passing negative values to np.sqrt
    Notice: even if this is a function of s, the function phaseshift is written
    as a function of sqrt(s). Inside 'omnes_rep_integrand' the square root is performed!
    """
    if s > s_th:
        res = omnes_above_threshold(s, s_th, phaseshift, *args, cut=cut)
    elif s <= s_th:
        res = omnes_below_threshold(s, s_th, phaseshift, *args)
    return res


omnes_function = np.vectorize(omnes_function_scalar)
