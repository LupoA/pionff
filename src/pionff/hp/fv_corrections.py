import numpy as np
from scipy.integrate import quad
from pionff.hp.core import matcal_T_pole, momenta_with_fixed_norm
from pionff.utils.amu_kernels import kernelTMR
from pionff.utils.debug_opt import timeit
from pionff.params import DEBUG_MODE


@timeit(DEBUG_MODE)
def correction_n_Ct(x0, n_mod, L, m_pi, phase_shift, *args):
    """
    Integral dk_3/2pi, from eq 2.4 of [https://arxiv.org/pdf/2004.03935.pdf]
    """

    def _integrand(k3):
        nL = n_mod * L
        res = (
            (np.cos(k3 * x0) / (24 * np.pi * nL))
            * matcal_T_pole(k3, n_mod, L, m_pi, phase_shift, *args)
            / (2 * np.pi)
        )
        return res

    value, _err = quad(
        lambda x: _integrand(x), 0, np.inf, epsabs=1.49e-08, epsrel=1.49e-08
    )
    return value * 2


@timeit(DEBUG_MODE)
def correction_amu_single_poisson_mode(
    x0cut, n_mod, L, m_muon, m_pi, phase_shift, *args
):
    """
    Single Poisson mode
    Eq. 2.9, using Eq. 2.10
    """

    def _integrand(x0):
        return kernelTMR(x0, m_muon) * correction_n_Ct(
            x0, n_mod, L, m_pi, phase_shift, *args
        )

    value, _err = quad(
        lambda x: _integrand(x), a=1e-10, b=x0cut, epsabs=1e-10, epsrel=1e-10
    )
    return value * momenta_with_fixed_norm(n_norm=n_mod)


@timeit(DEBUG_MODE)
def correction_amu(x0cut, L, m_muon, m_pi, phase_shift, *args):
    """
    First three Poisson modes
    """
    res = correction_amu_single_poisson_mode(
        x0cut, 1, L, m_muon, m_pi, phase_shift, *args
    )
    res += correction_amu_single_poisson_mode(
        x0cut, np.sqrt(2), L, m_muon, m_pi, phase_shift, *args
    )
    err = correction_amu_single_poisson_mode(
        x0cut, np.sqrt(3), L, m_muon, m_pi, phase_shift, *args
    )
    return res + err, err
