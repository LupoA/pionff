import numpy as np
from scipy.integrate import quad
from pionff.hp.core import matcal_T_pole, momenta_with_fixed_norm
from pionff.utils.amu_kernels import kernelTMR
from pionff.utils.windows import StandardWindows, sd_window
from pionff.params import gev_fm_conversion


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


def correction_amu_single_poisson_mode_window(
    window, x0_cut_dd, x0_cut_mll_hp, n_mod, L, m_muon, m_pi, phase_shift, *args
):
    """
    Single Poisson mode
    Eq. 2.9, using Eq. 2.10
    """

    window_t = StandardWindows()
    delta_gev = 0.15 * gev_fm_conversion

    def _integrand_full(x0):
        return (
            kernelTMR(x0, m_muon)
            * correction_n_Ct(x0, n_mod, L, m_pi, phase_shift, *args)
            * sd_window(x0, x0_cut_dd, delta_gev)
        )

    def _integrand_SD(x0):
        return (
            kernelTMR(x0, m_muon)
            * correction_n_Ct(x0, n_mod, L, m_pi, phase_shift, *args)
            * window_t.sd(x0, units_fm=False)
            * sd_window(x0, x0_cut_dd, delta_gev)
        )

    def _integrand_ID(x0):
        return (
            kernelTMR(x0, m_muon)
            * correction_n_Ct(x0, n_mod, L, m_pi, phase_shift, *args)
            * window_t.id(x0, units_fm=False)
            * sd_window(x0, x0_cut_dd, delta_gev)
        )

    def _integrand_LD(x0):
        return (
            kernelTMR(x0, m_muon)
            * correction_n_Ct(x0, n_mod, L, m_pi, phase_shift, *args)
            * window_t.ld(x0, units_fm=False)
            * sd_window(x0, x0_cut_dd, delta_gev)
        )

    def _integrand_OL(x0):
        return (
            kernelTMR(x0, m_muon)
            * correction_n_Ct(x0, n_mod, L, m_pi, phase_shift, *args)
            * window_t.ol(x0, units_fm=False)
        )

    if window == "full":
        value, _err = quad(
            lambda x: _integrand_full(x),
            a=1e-10,
            b=x0_cut_mll_hp,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return value * momenta_with_fixed_norm(n_norm=n_mod)

    elif window == "short_distance":
        value, _err = quad(
            lambda x: _integrand_SD(x),
            a=1e-10,
            b=x0_cut_mll_hp,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return value * momenta_with_fixed_norm(n_norm=n_mod)

    elif window == "intermediate_distance":
        value, _err = quad(
            lambda x: _integrand_ID(x),
            a=1e-10,
            b=x0_cut_mll_hp,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return value * momenta_with_fixed_norm(n_norm=n_mod)

    elif window == "long_distance":
        value, _err = quad(
            lambda x: _integrand_LD(x),
            a=1e-10,
            b=x0_cut_mll_hp,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return value * momenta_with_fixed_norm(n_norm=n_mod)

    elif window == "2.8-to-3.5":
        value, _err = quad(
            lambda x: _integrand_OL(x),
            a=1e-10,
            b=x0_cut_mll_hp,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return value * momenta_with_fixed_norm(n_norm=n_mod)

    else:
        raise ValueError(
            "Accepted values for 'window' are 'short_distance', 'intermediate_distance', 'long_distance', '2.8-to-3.5'."
        )


def correction_amu_window(
    window, x0_cut_dd, x0_cut_mll_hp, L, m_muon, m_pi, phase_shift, *args
):
    """
    First three Poisson modes
    """
    res = correction_amu_single_poisson_mode_window(
        window, x0_cut_dd, x0_cut_mll_hp, 1, L, m_muon, m_pi, phase_shift, *args
    )
    res += correction_amu_single_poisson_mode_window(
        window,
        x0_cut_dd,
        x0_cut_mll_hp,
        np.sqrt(2),
        L,
        m_muon,
        m_pi,
        phase_shift,
        *args,
    )
    err = correction_amu_single_poisson_mode_window(
        window,
        x0_cut_dd,
        x0_cut_mll_hp,
        np.sqrt(3),
        L,
        m_muon,
        m_pi,
        phase_shift,
        *args,
    )
    return res + err, err
