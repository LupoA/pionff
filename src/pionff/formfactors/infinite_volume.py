import numpy as np
from pionff.utils.kinematics import e_to_k_2particles
from scipy.integrate import quad
from pionff.utils.amu_kernels import kernelTMR


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

    def _integrate(t_val):
        return quad(lambda x: corr_integrand(x, t_val), (2 * m_pi), np.inf)[0]

    ct = np.vectorize(_integrate)(t)

    return ct


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
    )
    return amu
