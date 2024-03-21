import numpy as np
from pionff.utils.kinematics import e_to_k_2particles
from scipy.integrate import quad
from pionff.utils.amu_kernels import kernelTMR, kernelEQ
from pionff.utils.amu_kernels import _omega_2002_12347 as omega
from pionff.utils.debug_opt import timeit
from pionff.params import DEBUG_MODE


def spectral_density_iv(e, m_pi, absfpi, *absfpi_args):
    """
    HVP spectral density (infinite volume); dimensionless
    """
    k = e_to_k_2particles(e, m_pi)
    _res = (k / e) ** 3
    _res /= 6 * np.pi * np.pi
    absfp = absfpi(e, *absfpi_args)
    return _res * absfp * absfp


def corr_iv(t, m_pi, absfpi, *absfpi_args):
    """
    \int_0^inf rho(E) E^2 exp(-tE) ::: has dimension [E]^3
    """

    def corr_integrand(e, t_val):
        """
        rho(E) E^2 exp(-tE) ::: has dimension [E]^2
        """
        return (
            spectral_density_iv(e, m_pi, absfpi, *absfpi_args)
            * e
            * e
            * np.exp(-t_val * e)
        )

    result, _ = quad(
        corr_integrand, 2 * m_pi, np.inf, args=(t,), epsabs=1e-12, epsrel=1e-12
    )
    return result


@timeit(DEBUG_MODE)
def a_mu_iv(mass_muon, m_pi, absfpi, *absfpi_args, x0min=0, x0max=np.inf):
    """
    a_mu = \int_min^max dx0 C(x0) K(x0)
    """

    def integrand_for_amu(x0, mass_muon, m_pi, absfpi, *_absfpi_args):
        return corr_iv(x0, m_pi, absfpi, *_absfpi_args) * kernelTMR(
            x0, mass_muon=mass_muon
        )

    amu, _err = quad(
        lambda x: integrand_for_amu(x, mass_muon, m_pi, absfpi, *absfpi_args),
        x0min,
        x0max,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return amu


@timeit(DEBUG_MODE)
def a_mu_from_rho_iv(mass_muon, m_pi, absfpi, *absfpi_args, x0min=0, x0max=np.inf):
    """
    a_mu = \int dE E^2 rho(E) \int dQ^2/m^2 omega(Q^2/m^2) kernelEQ(E,Q^2)
    """

    def _integrate_qsq(e):
        def _integrand_qsq(qsq, e):
            return omega(qsq / mass_muon**2) * kernelEQ(e, qsq, mass_muon, x0min, x0max)

        return quad(
            lambda x: _integrand_qsq(x, e), 0, np.inf, epsabs=1e-12, epsrel=1e-12
        )[0]

    def _integrand(e):
        return (
            spectral_density_iv(e, m_pi, absfpi, *absfpi_args)
            * e
            * e
            * _integrate_qsq(e)
        )

    return quad(lambda x: _integrand(x), 2 * m_pi, np.inf, epsabs=1e-12, epsrel=1e-12)[
        0
    ]
