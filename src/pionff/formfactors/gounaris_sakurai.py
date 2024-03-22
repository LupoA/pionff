#   Gournaris Sakurai parametrisation of the pion FF
import numpy as np
from pionff.utils.kinematics import e_to_k_2particles
from scipy.integrate import quad
from pionff.utils.amu_kernels import kernelTMR


def k_rho_function(m_pi, m_rho):
    return e_to_k_2particles(m_rho, m_pi)


def gamma_rho_const(m_pi, m_rho, g_ppr):
    gfact = (g_ppr * g_ppr) / (6 * np.pi)
    k_rho = k_rho_function(m_pi, m_rho)
    return gfact * (k_rho * k_rho * k_rho) / (m_rho * m_rho)


def g_from_gamma(m_pi, m_rho, gamma_rho):
    k_rho = k_rho_function(m_pi, m_rho)
    res = m_rho * m_rho / (k_rho**3)
    res *= gamma_rho * np.pi * 6
    return np.sqrt(res)


def gamma_width(
    e, m_pi, m_rho, g_ppr
):  #   eq. 115 of https://arxiv.org/pdf/2002.12347.pdf
    gamma_rho = gamma_rho_const(m_pi, m_rho, g_ppr)
    k = e_to_k_2particles(e, m_pi)
    k_rho = k_rho_function(m_pi, m_rho)
    res = k / k_rho
    res = res * res * res
    res *= m_rho / e
    return res * gamma_rho


def h_function(e, m_pi):  #   eq. 117 of https://arxiv.org/pdf/2002.12347.pdf
    k = e_to_k_2particles(e, m_pi)
    res = 2 * k / (np.pi * e)
    res *= np.log((e + (2 * k)) / (2 * m_pi))
    return res


def h_prime(e, m_pi):
    res = np.log((e + np.sqrt((e * e) - (4 * m_pi * m_pi))) / (2 * m_pi))
    res *= 1 / np.sqrt((e * e) - (4 * m_pi * m_pi))
    res *= 4 * m_pi * m_pi
    res /= e * e * np.pi
    return res + (1 / (np.pi * e))


def h_prime_ee(e, m_pi):  #   derivative of 117 wrt E^2 so must multiply by 2*E
    k = e_to_k_2particles(e, m_pi)
    res = 1 / (2 * e * e)
    res += ((m_pi * m_pi) / (k * e * e * e)) * np.log((e + (2 * k)) / (2 * m_pi))
    return res / np.pi


def mSquare(e, m_pi, m_rho, g_ppr):
    gamma_rho = gamma_rho_const(m_pi=m_pi, m_rho=m_rho, g_ppr=g_ppr)
    k = e_to_k_2particles(e, m_pi)
    k_rho = k_rho_function(m_pi=m_pi, m_rho=m_rho)
    res = m_rho**2
    _term = gamma_rho * res / (k_rho**3)
    _braket1 = (h_function(e, m_pi) - h_function(e=m_rho, m_pi=m_pi)) * k * k
    _braket2 = ((e**2) - (m_rho**2)) * h_prime(e=m_rho, m_pi=m_pi)
    _braket2 *= k_rho * k_rho * 0.5
    _braket2 /= m_rho
    res += _term * (_braket1 - _braket2)
    return res


def mSqZero(m_pi, m_rho, g_ppr):
    gamma_rho = gamma_rho_const(m_pi=m_pi, m_rho=m_rho, g_ppr=g_ppr)
    k_rho = k_rho_function(m_pi=m_pi, m_rho=m_rho)
    res = m_rho * m_rho * gamma_rho / (k_rho * k_rho * k_rho)
    _term = h_prime(e=m_rho, m_pi=m_pi) * k_rho * k_rho * m_rho * 0.5
    _term -= m_pi * m_pi / np.pi
    _term += m_pi * m_pi * h_function(e=m_rho, m_pi=m_pi)
    res *= _term
    res += m_rho * m_rho
    return res


######################################


def cmplx_Fpi(e, m_pi, m_rho, g_ppr):
    res = mSqZero(m_pi=m_pi, m_rho=m_rho, g_ppr=g_ppr)
    den = (
        mSquare(e, m_pi, m_rho, g_ppr)
        - e**2
        - (1j * gamma_width(e, m_pi, m_rho, g_ppr) * m_rho)
    )
    return res / den


def absFpi(e, m_pi, m_rho, g_ppr):
    _a = mSqZero(m_pi=m_pi, m_rho=m_rho, g_ppr=g_ppr)
    _b = mSquare(e, m_pi, m_rho, g_ppr) - e**2
    _c = gamma_width(e, m_pi, m_rho, g_ppr) * m_rho
    res = _a / np.sqrt(_b**2 + _c**2)
    return res


def cot_delta_11(e, m_pi, m_rho, g_ppr):
    """
    a discontinuous function
    """
    _b = mSquare(e, m_pi, m_rho, g_ppr) - e**2
    _c = gamma_width(e, m_pi, m_rho, g_ppr) * m_rho
    return _b / _c


def argFpi(e, m_pi, m_rho, g_ppr):  #   aka delta
    """
    manually made continuous by shifting by pi when e > m_rho
    """
    _b = mSquare(e, m_pi, m_rho, g_ppr) - e**2
    _c = gamma_width(e, m_pi, m_rho, g_ppr) * m_rho
    res = _c / _b
    res = np.arctan(res)
    if isinstance(res, np.ndarray):
        res[e > m_rho] += np.pi
    elif e > m_rho:
        res += np.pi
    return res


def phase_shift(e, m_pi, m_rho, g_ppr):
    return argFpi(e, m_pi, m_rho, g_ppr)


def spectral_density(e, m_pi, m_rho, g_ppr):
    """
    HVP spectral density (infinite volume); dimensionless
    """
    k = e_to_k_2particles(e, m_pi)
    _res = (k / e) ** 3
    _res /= 6 * np.pi * np.pi
    absfp = absFpi(e, m_pi=m_pi, m_rho=m_rho, g_ppr=g_ppr)
    return _res * absfp * absfp


def gs_corr(t, m_pi, m_rho, g_ppr):
    """
    \int_0^inf rho(E) E^2 exp(-tE) ::: has dimension [E]^3
    """

    def gs_corr_integrand(e):
        """
        rho(E) E^2 exp(-tE) ::: has dimension [E]^2
        """
        return spectral_density(e, m_pi, m_rho, g_ppr) * e * e * np.exp(-t * e)

    ct, _err = quad(lambda x: gs_corr_integrand(x), (2 * m_pi), np.inf)
    return ct


def gs_amu(mass_muon, m_pi, m_rho, g_ppr, x0min=0, x0max=np.inf):
    def integrand_for_amu(x0, mass_muon, m_pi, m_rho, g_ppr):
        return gs_corr(x0, m_pi, m_rho, g_ppr) * kernelTMR(x0, mass_muon=mass_muon)

    amu, _err = quad(
        lambda x: integrand_for_amu(x, mass_muon, m_pi, m_rho, g_ppr), x0min, x0max
    )
    return amu
