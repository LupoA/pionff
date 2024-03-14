import numpy as np
from scipy.integrate import quad


def omnes_rep_integrand(sp, s, phaseshift, *args):
    """
    returns delta(s') / s' (s-s') to be integrated in s'
    args: positional arguments of phaseshift
    """
    res = phaseshift(np.sqrt(sp), *args)
    res /= sp * (s - sp)
    return res


def omnes_below_threshold(s, s_th, phaseshift, *args):
    """
    args: positional arguments of phaseshift
    s_th = 4 m_pi^2
    """
    res, _err = quad(omnes_rep_integrand, s_th, np.inf, args=(s, phaseshift, *args))
    res *= s / np.pi
    return np.exp(-res)


def omnes_above_threshold(s, s_th, phaseshift, *args):
    """
    args: positional arguments of phaseshift
    s_th = 4 m_pi^2
    Integral formally divergent: point around the singularity is removed
    within an interval 2e-7
    """
    res_a, _err = quad(
        omnes_rep_integrand, s_th, s - (1e-7), args=(s, phaseshift, *args)
    )
    res_b, _err = quad(
        omnes_rep_integrand, s + (1e-7), np.inf, args=(s, phaseshift, *args)
    )
    res = res_a + res_b
    res *= s / np.pi
    return np.exp(-res)


def omnes_function_scalar(s, s_th, phaseshift, *args):
    """
    splitting is needed to avoid passing negative values to np.sqrt
    """
    if s > s_th:
        res = omnes_above_threshold(s, s_th, phaseshift, *args)
    elif s <= s_th:
        res = omnes_below_threshold(s, s_th, phaseshift, *args)
    return res


omnes_function = np.vectorize(omnes_function_scalar)

#   #   #   #   specialised functions for gounaris sakurai


def omega_GS_below_threshold(s, m_pi, m_rho, g_ppr):
    from pionff.formfactors.gounaris_sakurai import argFpi as deltaGS

    res, _err = quad(
        omnes_rep_integrand,
        (2 * m_pi) ** 2,
        np.inf,
        args=(s, deltaGS, m_pi, m_rho, g_ppr),
    )
    res *= s / np.pi
    return np.exp(-res)


def omega_GS_above_threshold(s, m_pi, m_rho, g_ppr, cut=1e-8):
    from pionff.formfactors.gounaris_sakurai import argFpi as deltaGS

    res_a, _err = quad(
        omnes_rep_integrand,
        a=(2 * m_pi) ** 2,
        b=s - cut,
        args=(s, deltaGS, m_pi, m_rho, g_ppr),
    )
    res_b, _err = quad(
        omnes_rep_integrand, a=s + cut, b=np.inf, args=(s, deltaGS, m_pi, m_rho, g_ppr)
    )
    res = res_a + res_b
    res *= s / np.pi
    return np.exp(-res)


def omega_GS(s, m_pi, m_rho, g_ppr):
    if s > (4 * m_pi * m_pi):
        res = omega_GS_above_threshold(s, m_pi, m_rho, g_ppr, cut=1e-6)
    elif s <= (4 * m_pi * m_pi):
        res = omega_GS_below_threshold(s, m_pi, m_rho, g_ppr)
    return res
