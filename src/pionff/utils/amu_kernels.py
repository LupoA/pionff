import numpy as np
from pionff.params import gamma_eul_masc
from scipy.integrate import quad

"""
two representations of the kernel are provided
"""


def integrand_dv(v, par_t):
    """
    from eq. 2.10 of hp [2004.03935]
    """
    _term = np.sqrt((v * v) + 4)
    return np.exp(-par_t * _term) / (_term**3)


def integral_dv(par_t):
    """
    from eq. 2.10 of hp [2004.03935]
    """
    value, _err = quad(lambda x: integrand_dv(x, par_t), 0, np.inf)
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
def omega_2002_12347(r):
    """
    from [BMW 2020]
    """
    _term = np.sqrt(r * (r + 4))
    return ((r + 2 - _term) ** 2) / _term


def integrand_2002_12347(Qsq, mass_muon, x0):
    """
    from [BMW 2020]
    """
    res = omega_2002_12347(Qsq / (mass_muon * mass_muon))
    return (res * ((x0 * x0) - (4 / Qsq) * (np.sin(np.sqrt(Qsq) * x0 * 0.5) ** 2))) / (
        mass_muon * mass_muon
    )


def kernel_2002_12347(x0, mass_muon, alpha=(1 / 137)):
    """
    from [BMW 2020]
    """
    value, _err = quad(
        lambda x: integrand_2002_12347(x, mass_muon=mass_muon, x0=x0), 0, np.inf
    )
    return value * alpha * alpha


def kernelTMR(x0, mass_muon, alpha=(1 / 137)):
    """
    Selects kernel from [BMW 2020]. The other is used to cross-check.
    Already includes factors of alpha and the mass of the muon
    Has dimensions [E]^{-2}
    """
    return kernel_2002_12347(x0=x0, mass_muon=mass_muon, alpha=alpha)
