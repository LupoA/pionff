import numpy as np
from scipy.integrate import quad


def _z_factor(p3, msq, k3sq, sign: int):
    """
    Z_{\pm} factor appearing eq 2.26 of [https://arxiv.org/pdf/2004.03935.pdf]
    """
    p3sq = p3 * p3
    _term1 = msq + p3sq + (k3sq * 0.25)
    _term2 = sign * 0.5 * np.sqrt((_term1**2) - (p3sq * k3sq))
    _term1 *= 0.5
    return np.sqrt(_term1 + _term2)


def _zeta_226(k3, n_mod, L, msq):  # eq 2.26
    """
    auxiliary function denoted by zeta in [https://arxiv.org/pdf/2004.03935.pdf]
    Here with the representation given by Eq. 2.26
    """

    def _integrand(p3):
        """
        integrand for eq 2.26 of [https://arxiv.org/pdf/2004.03935.pdf]
        """
        nL = n_mod * L
        _term1 = np.exp(-nL * _z_factor(p3, msq, k3sq, sign=+1)) / _z_factor(
            p3, msq, k3sq, sign=+1
        )
        _term2 = np.sinh(nL * _z_factor(p3, msq, k3sq, sign=-1)) / (
            2 * _z_factor(p3, msq, k3sq, sign=-1)
        )
        return (_term1 * _term2) / (2 * np.pi)

    k3sq = k3**2
    result, _err = quad(lambda x: _integrand(x), 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    return result * 2


def _zeta_c15(
    k3, msq, n_mod, L
):  # eq 2.26 integral from -inf to inf -> 2x (0,inf) for speed
    """
    Another definition for zeta given in Eq. c15 of [https://arxiv.org/pdf/2004.03935.pdf]
    """

    def _integrand(p3):
        _minus = np.sqrt(msq + ((p3 - (k3 / 2)) ** 2))
        _plus = np.sqrt(msq + ((p3 + (k3 / 2)) ** 2))
        return (np.exp(-n_mod * L * _minus) - np.exp(-n_mod * L * _plus)) / (
            (2 * k3 * p3) * 2 * np.pi
        )

    result, _err = quad(lambda x: _integrand(x), 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    return result * 2


def zeta(k3, msq, n_mod, L):
    return _zeta_c15(k3, msq, n_mod, L)
