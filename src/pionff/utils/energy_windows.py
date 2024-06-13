import numpy as np
from pionff.params import gev_fm_conversion, mass_muon_GeV
from pionff.utils.amu_kernels import kernelTMR
from scipy.integrate import quad
from pionff.utils.windows import sd_window, id_window, ld_window


def sd_window_R(e, t0, delta=0.15 * gev_fm_conversion, m_muon=mass_muon_GeV):
    def _integrand_sd(x0):
        return np.exp(-x0 * e) * kernelTMR(x0, m_muon) * sd_window(x0, t0, delta)

    res, _ = quad(_integrand_sd, 0, np.inf)
    return res * e * e


def id_window_R(e, t0, t1, delta=0.15 * gev_fm_conversion, m_muon=mass_muon_GeV):
    def _integrand_id(x0):
        return np.exp(-x0 * e) * kernelTMR(x0, m_muon) * id_window(x0, t0, t1, delta)

    res, _ = quad(_integrand_id, 0, np.inf)
    return res * e * e


def ld_window_R(e, t1, delta=0.15 * gev_fm_conversion, m_muon=mass_muon_GeV):
    def _integrand_ld(x0):
        return np.exp(-x0 * e) * kernelTMR(x0, m_muon) * ld_window(x0, t1, delta)

    res, _ = quad(_integrand_ld, 0, np.inf)
    return res * e * e
