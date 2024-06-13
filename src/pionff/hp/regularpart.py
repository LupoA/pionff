import numpy as np
from scipy.integrate import quad
from pionff.utils.amu_kernels import _omega_2002_12347 as omega_function
from pionff.hp.core import momenta_with_fixed_norm
from pionff.utils.windows import StandardWindows, sd_window, id_window
from pionff.params import gev_fm_conversion


def isolated_p3_integral(m_pi, n_mod, L):
    def _integrandp3(p3):
        msq = m_pi * m_pi
        psq = p3 * p3
        _res = np.exp(-n_mod * L * np.sqrt(msq + psq)) / (24 * n_mod * np.pi * L)
        _res /= 2 * np.pi
        return _res * 2

    result, _err = quad(
        lambda x: _integrandp3(x), 0, np.inf, epsabs=1e-10, epsrel=1e-10
    )
    return result


def d4_treg(ksq, m_pi, fpi=0.132):
    """fourth derivative of NLO chpt Treg"""

    mm = m_pi * m_pi
    _root = 4 * mm / ksq
    _root = np.sqrt(_root + 1)

    _left = -2 * ksq * _root
    res = 4 * ksq * ksq * ksq * ksq
    res += 93 * ksq * ksq * ksq * mm
    res += 654 * ksq * ksq * mm * mm
    res += 3808 * ksq * mm * mm * mm
    res += 5376 * mm * mm * mm * mm
    _left *= res

    _right = 48 * mm * mm * np.arctanh(1 / _root)
    res = 15 * ksq * ksq * ksq
    res += 245 * ksq * ksq * mm
    res += 784 * ksq * mm * mm
    res += 896 * mm * mm * mm
    _right *= res

    res = (ksq + (4 * mm)) * ksq
    res *= res * res
    res = (_left + _right) / res
    res /= _root
    res /= 6 * np.pi * np.pi * fpi * fpi

    return res


def integrate_k(x0, m_pi):
    """
    Integrate cos(x0 k) with the d^4 dk^4 T_reg(k) dk
    """

    def _integrandk(k):
        return np.cos(x0 * k) * d4_treg(k * k, m_pi=m_pi) / (2 * np.pi)

    res, _ = quad(lambda x: _integrandk(x), 1e-10, np.inf, epsabs=1e-10, epsrel=1e-10)
    return res * 2


def integrate_x0(q, m_pi, x0_min, x0_max):
    def _integrandx0(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0)
        _res *= _res
        _res /= q
        _res = (1 / (x0 * x0)) - _res
        _res *= integrate_k(x0, m_pi)
        return _res

    if x0_min == 0:
        x0_min = 1e-10

    res, _ = quad(lambda x: _integrandx0(x), x0_min, x0_max, epsabs=1e-10, epsrel=1e-10)
    print("integral in x0 returns ", res)
    return res


"""
def amu_reg_singlePoissoneMode(n_mod, L, m_muon, m_pi, x0_min, x0_max, alpha=1/137):

    m_muon_sq = m_muon * m_muon

    def _integrandQ(qsq):

        _q = np.sqrt(qsq)
        _res = omega_function(qsq / m_muon_sq) * integrate_x0(_q, m_pi, x0_min, x0_max)
        return _res

    res, _ = quad(lambda x: _integrandQ(x), 0, np.inf, epsabs=1e-10, epsrel=1e-10)
    res *= alpha * alpha / m_muon_sq
    res *= isolated_p3_integral(m_pi, n_mod, L)
    print('poisson mode returns', res)
    return res

def amu_reg(L, m_muon, m_pi, x0_min, x0_max, alpha=1/137):
    res = amu_reg_singlePoissoneMode(1, L, m_muon, m_pi, x0_min, x0_max, alpha=1 / 137)
    res += amu_reg_singlePoissoneMode(np.sqrt(2), L, m_muon, m_pi, x0_min, x0_max, alpha=1/137)
    res += amu_reg_singlePoissoneMode(np.sqrt(3), L, m_muon, m_pi, x0_min, x0_max, alpha=1/137)
    return res
"""

############# other way around
# epsabs=1e-10, epsrel=1e-10


def do_x0_first_windows(q, k, x0min, x0max, window):
    window_t = StandardWindows()
    delta_gev = 0.15 * gev_fm_conversion

    def _integrand_00_04(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0 * q)
        _res *= _res
        _res = (1 / (x0 * x0)) - _res
        _res *= np.cos(x0 * k)
        _res *= window_t.sd(x0, units_fm=False)
        return _res

    def _integrand_04_10(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0 * q)
        _res *= _res
        _res = (1 / (x0 * x0)) - _res
        _res *= np.cos(x0 * k)
        _res *= window_t.id(x0, units_fm=False)
        return _res

    def _integrand_10_28(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0 * q)
        _res *= _res
        _res = (1 / (x0 * x0)) - _res
        _res *= np.cos(x0 * k)
        _res *= id_window(x0, 1 * gev_fm_conversion, 2.8 * gev_fm_conversion, delta_gev)
        return _res

    def _integrand_00_28(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0 * q)
        _res *= _res
        _res = (1 / (x0 * x0)) - _res
        _res *= np.cos(x0 * k)
        _res *= sd_window(x0, 2.8 * gev_fm_conversion, delta_gev)
        return _res

    def _integrand_15_19(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0 * q)
        _res *= _res
        _res = (1 / (x0 * x0)) - _res
        _res *= np.cos(x0 * k)
        _res *= id_window(
            x0, 1.5 * gev_fm_conversion, 1.9 * gev_fm_conversion, delta_gev
        )
        return _res

    def _integrand_28_35(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0 * q)
        _res *= _res
        _res = (1 / (x0 * x0)) - _res
        _res *= np.cos(x0 * k)
        _res *= id_window(
            x0, 2.8 * gev_fm_conversion, 3.5 * gev_fm_conversion, delta_gev
        )
        return _res

    if window == "00-28":
        res, _ = quad(_integrand_00_28, a=x0min, b=x0max, epsabs=1e-5, epsrel=1e-5)
    elif window == "00-04":
        res, _ = quad(_integrand_00_04, a=x0min, b=x0max, epsabs=1e-5, epsrel=1e-5)
    elif window == "04-10":
        res, _ = quad(_integrand_04_10, a=x0min, b=x0max, epsabs=1e-5, epsrel=1e-5)
    elif window == "10-28":
        res, _ = quad(_integrand_10_28, a=x0min, b=x0max, epsabs=1e-5, epsrel=1e-5)
    elif window == "15-19":
        res, _ = quad(_integrand_15_19, a=x0min, b=x0max, epsabs=1e-5, epsrel=1e-5)
    elif window == "28-35":
        res, _ = quad(_integrand_28_35, a=x0min, b=x0max, epsabs=1e-5, epsrel=1e-5)
    else:
        raise ValueError("Invalid window")

    return res


def do_x0_first(q, k, x0min, x0max):
    def _integrand_do_x0_first(x0):
        _res = 2 * np.sin(q * x0 / 2) / (x0 * x0 * q)
        _res *= _res
        _res = (1 / (x0 * x0)) - _res
        _res *= np.cos(x0 * k)
        return _res

    res, _ = quad(_integrand_do_x0_first, a=x0min, b=x0max, epsabs=1e-5, epsrel=1e-5)
    return res


def do_k_second(q, m_pi, x0min, x0max, window):
    def _integrand_do_k_second(k):
        return (
            d4_treg(k * k, m_pi)
            * do_x0_first_windows(q, k, x0min, x0max, window)
            / (2 * np.pi)
        )

    res, _ = quad(_integrand_do_k_second, a=0, b=20, epsabs=1e-5, epsrel=1e-5)
    return res * 2


def do_Qsq_last(m_muon, m_pi, x0min, x0max, window, alpha=1 / 137):
    m_muon_sq = m_muon * m_muon

    def _integrand_do_Qsq_last(qsq):
        return omega_function(qsq / m_muon_sq) * do_k_second(
            np.sqrt(qsq), m_pi, x0min, x0max, window
        )

    res, _ = quad(_integrand_do_Qsq_last, a=0, b=np.inf)
    res *= alpha * alpha / m_muon_sq
    return res


def amu_reg_singlePoissoneMode(
    n_mod, L, m_muon, m_pi, x0_min, x0_max, window, alpha=1 / 137
):
    res = (
        do_Qsq_last(m_muon, m_pi, x0_min, x0_max, window, alpha)
        * isolated_p3_integral(m_pi, n_mod, L)
        * momenta_with_fixed_norm(n_norm=n_mod)
    )
    # print('poisson mode returns', res)
    return res


def amu_reg(L, m_muon, m_pi, x0_min, x0_max, window, alpha=1 / 137):
    res = amu_reg_singlePoissoneMode(1, L, m_muon, m_pi, x0_min, x0_max, window, alpha)
    res += amu_reg_singlePoissoneMode(
        np.sqrt(2), L, m_muon, m_pi, x0_min, x0_max, window, alpha
    )
    res += amu_reg_singlePoissoneMode(
        np.sqrt(3), L, m_muon, m_pi, x0_min, x0_max, window, alpha
    )

    return res
