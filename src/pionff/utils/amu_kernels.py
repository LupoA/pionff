import numpy as np
from pionff.params import gamma_eul_masc
from scipy.integrate import quad

"""
two representations of the kernel are provided
"""


def _integrand_dv(v, par_t):
    """
    from eq. 2.10 of hp [2004.03935]
    """
    _term = np.sqrt((v * v) + 4)
    return np.exp(-par_t * _term) / (_term**3)


def integral_dv(par_t):
    """
    from eq. 2.10 of hp [2004.03935]
    """
    value, _err = quad(
        lambda x: _integrand_dv(x, par_t), 0, np.inf, epsabs=1e-12, epsrel=1e-12
    )
    return value


def kernelK_2004_03935(x0, mass_muon, alpha=(1 / 137)):
    """
    Eq. 2.10 of hp in full [2004.03935]
    """
    from scipy.special import kn

    _t = x0 * mass_muon
    res = (
        (_t * _t)
        - (2 * np.pi * _t)
        + (8 * gamma_eul_masc)
        - 2
        + (4 / (_t * _t))
        + (8 * np.log(_t))
        - (8 * kn(1, 2 * _t) / _t)
        - (8 * integral_dv(_t))
    )
    return res * 2 * alpha * alpha / (mass_muon * mass_muon)


#   Following Eq. 3 of https://arxiv.org/pdf/2002.12347.pdf (BMW 2020)
def _omega_2002_12347(r):
    """
    from [BMW 2020]
    """
    _term = np.sqrt(r * (r + 4))
    return ((r + 2 - _term) ** 2) / _term


def _integrand_2002_12347(Qsq, mass_muon, x0):
    """
    from [BMW 2020]
    """
    res = _omega_2002_12347(Qsq / (mass_muon * mass_muon))
    return (res * ((x0 * x0) - (4 / Qsq) * (np.sin(np.sqrt(Qsq) * x0 * 0.5) ** 2))) / (
        mass_muon * mass_muon
    )


def kernel_2002_12347(x0, mass_muon, alpha=(1 / 137)):
    """
    from [BMW 2020]
    """
    value, _err = quad(
        lambda x: _integrand_2002_12347(x, mass_muon=mass_muon, x0=x0),
        0,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=100,
    )
    return value * alpha * alpha


def kernelTMR(x0, mass_muon, alpha=(1 / 137)):
    """
    Selects kernel from [BMW 2020]. The other is used to cross-check.
    Already includes factors of alpha and the mass of the muon
    Has dimensions [E]^{-2}
    """
    return kernel_2002_12347(x0=x0, mass_muon=mass_muon, alpha=alpha)


def _kernelEQ(e, qsq, xmin=0, xmax=np.inf):
    """
    Usage:
        For computations of a_mu from the spectral density:
            a_mu = alpha^2 \int dE E^2 rho(E) \int dQ^2 / m_mu^2 omega(Q^2/m^2) _kernelEQ(E,Q^2)
        where _kernelEQ has the integral over x0 being performed analytically
            _kernelEQ(E,Q^2) = int dx0 [t^2 - 4/Q^2 Sin^2{Qt/2}] Exp[-t*E]
    Returns:
        _kernelEQ(E,Q^2)
    """
    assert xmin >= 0
    if xmin == 0 and xmax == np.inf:
        result = 2 * qsq / ((e**5) + ((e**3) * qsq))
        return result
    elif xmin != 0 and xmax == np.inf:
        term1 = (1 / (e**3 * qsq * (e**2 + qsq))) * np.exp(-e * xmin)
        term2 = (e**2 + qsq) * (
            2 * qsq + 2 * e * qsq * xmin + e**2 * (-2 + qsq * xmin**2)
        )
        term3 = 2 * e**4 * np.cos(np.sqrt(qsq) * xmin)
        term4 = -2 * e**3 * np.sqrt(qsq) * np.sin(np.sqrt(qsq) * xmin)
        result = term1 * (term2 + term3 + term4)
        return result
    else:
        raise ValueError(
            "x_max < infinity not yet implemented analytically. Consider solving it numerically, or implement it."
        )


def kernelEQ(e, qsq, mass_muon, xmin=0, xmax=np.inf, alpha=(1 / 137)):
    """
    Multiplies _kernelEQ by alpha^2 / m_muon^2
    Result has dimensions [E]^{-5}
    """
    return _kernelEQ(e, qsq, xmin, xmax) * alpha * alpha / (mass_muon * mass_muon)
