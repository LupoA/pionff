import numpy as np
from scipy.integrate import quad
from pionff.formfactors.omnes import omnes_below_threshold


def _z_factor(p3, m_pi_sq, k3sq, sign: int):
    """
    Z_{\pm} factor appearing eq 2.26 of [https://arxiv.org/pdf/2004.03935.pdf]
    """
    p3sq = p3 * p3
    _term1 = m_pi_sq + p3sq + (k3sq * 0.25)
    _term2 = sign * 0.5 * np.sqrt((_term1**2) - (p3sq * k3sq))
    _term1 *= 0.5
    return np.sqrt(_term1 + _term2)


def _zeta_226(k3, n_mod, L, m_pi_sq):  # eq 2.26
    """
    auxiliary function denoted by zeta in [https://arxiv.org/pdf/2004.03935.pdf]
    Here with the representation given by Eq. 2.26
    """

    def _integrand(p3):
        """
        integrand for eq 2.26 of [https://arxiv.org/pdf/2004.03935.pdf]
        """
        nL = n_mod * L
        _term1 = np.exp(-nL * _z_factor(p3, m_pi_sq, k3sq, sign=+1)) / _z_factor(
            p3, m_pi_sq, k3sq, sign=+1
        )
        _term2 = np.sinh(nL * _z_factor(p3, m_pi_sq, k3sq, sign=-1)) / (
            2 * _z_factor(p3, m_pi_sq, k3sq, sign=-1)
        )
        return (_term1 * _term2) / (2 * np.pi)

    k3sq = k3**2
    result, _err = quad(lambda x: _integrand(x), 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    return result * 2


def _zeta_c15(
    k3, m_pi_sq, n_mod, L
):  # eq 2.26 integral from -inf to inf -> 2x (0,inf) for speed
    """
    Another definition for zeta given in Eq. c15 of [https://arxiv.org/pdf/2004.03935.pdf]
    """

    def _integrand(p3):
        _minus = np.sqrt(m_pi_sq + ((p3 - (k3 / 2)) ** 2))
        _plus = np.sqrt(m_pi_sq + ((p3 + (k3 / 2)) ** 2))
        return (np.exp(-n_mod * L * _minus) - np.exp(-n_mod * L * _plus)) / (
            (2 * k3 * p3) * 2 * np.pi
        )

    result, _err = quad(lambda x: _integrand(x), 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    return result * 2


def zeta(k3, msq, n_mod, L):
    return _zeta_c15(k3, msq, n_mod, L)


def matcal_T_pole_wAlpha(k3, n_mod, L, m_pi, alpha_V, phase_shift, *args):
    """
    Eq. 2.24 of [https://arxiv.org/pdf/2004.03935.pdf]
    """
    # e = k_to_E_2particles(m_pi, k3)
    k3sq = k3**2
    msq = m_pi**2
    # s = e**2

    def _absfpi(_k3sq):
        return omnes_below_threshold(_k3sq, 2 * m_pi, np.inf, phase_shift, *args)

    return (
        2
        * ((4 * msq) + k3sq)
        * _absfpi(-k3sq)
        * _absfpi(-k3sq)
        * zeta(k3, msq, n_mod, L)
    )


def momenta_with_fixed_norm(n_norm):
    if n_norm == 1:
        return 6
    if n_norm == np.sqrt(2):
        return 12
    if n_norm == np.sqrt(3):
        return 8
    else:
        ValueError("|n| must be either: 1, sqrt2, sqrt3")
