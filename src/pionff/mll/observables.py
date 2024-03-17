from pionff.mll.fv_amplitudes import get_amplitudes
from pionff.mll.fv_energies import get_energies
from pionff.utils.amu_kernels import kernelTMR
from scipy.integrate import quad
import logging
import numpy as np


def solve_ll(n_states, L, m_pi, phase_shift, absfpi, phase_args=(), absfpi_args=()):
    """
    Get finite volume energies and matrix elements
    """
    logging.info("solve_ll ::: Computing finite volume energies and matrix elements")
    logging.info("States: ", n_states, "L: ", L, "m_pi: ", m_pi, " mL: ", m_pi * L)
    a_n_sq_array = np.zeros(n_states)
    e_n_array = np.zeros(n_states)
    for n in range(n_states):
        a_n_sq_array[n], _, _ = get_amplitudes(
            n + 1, L, m_pi, phase_shift, absfpi, phase_args, absfpi_args
        )
        _, _, _, e_n_array[n] = get_energies(L, n + 1, m_pi, phase_shift, *phase_args)
    logging.info("solve_ll ::: Energies (units depend on inputs)", e_n_array)
    logging.info(
        "solve_ll ::: Squared matrix elements (units depend on inputs)", a_n_sq_array
    )
    return e_n_array, a_n_sq_array


def mll_corr(t, e_n, a_n):
    """
    finite-volume correlator in time-momentum representation
    units of a_n determines units of result, which is [3] for the HVP
    result is divided by 3 in order to reproduce eq. 120 of [BMW 2020]
    """
    n_states = len(e_n)
    assert n_states == len(a_n)
    corr = 0
    for _n in range(n_states):
        corr += a_n[_n] * np.exp(-t * e_n[_n])
    err = a_n[-1] * np.exp(-t * e_n[-1])
    return corr / 3, err / 3


def mll_ssd(e, sigma, e_n, a_n):
    """
    finite-volume gaussian-smeared spectral density
    """
    n_states = len(e_n)
    assert len(a_n) == n_states
    e = np.array(e)
    srho = np.zeros_like(e)
    from scipy.stats import norm

    for n in range(n_states):
        srho += a_n[n] * norm.pdf(e, loc=e_n[n], scale=sigma)
    # err = a_n[-1] * norm.pdf(e, loc=e_n[-1], scale=sigma)
    return srho / 3


def mll_amu(mass_muon, e_n, asq_n, x0min=0, x0max=np.inf):
    """
    return a_mu = int_{x0 min}^{x0 max} K(x0) C(x0)
    """

    def _integrand_central(x0, asq_n, e_n, mass_muon):
        """
        K(x_0) C(x_0) and K(x_0) delta_C(x_0)
        """
        corr_value, err = mll_corr(x0, e_n, asq_n)
        return corr_value * kernelTMR(x0, mass_muon=mass_muon), err * kernelTMR(
            x0, mass_muon=mass_muon
        )

    amu, _ = quad(
        lambda x: _integrand_central(x, asq_n, e_n, mass_muon)[0], x0min, x0max
    )
    amuERR, _ = quad(
        lambda x: _integrand_central(x, asq_n, e_n, mass_muon)[1], x0min, x0max
    )
    return amu, amuERR
