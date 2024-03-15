# from pionff.params import mass_pi0_GeV, mass_rho0_GeV, width_rho0_neutral, g_ppr_0, Lref_fm, gev_fm_conversion
from pionff.formfactors.gounaris_sakurai import g_from_gamma, argFpi, absFpi
from pionff.mll.fv_amplitudes import get_amplitudes
import numpy as np


def test_amplitudes():
    """
    test against table 1 of
    https://arxiv.org/pdf/1306.2532.pdf
    """
    m_pi_ref_gev = 0.13957
    m_rho_ref_gev = 0.773
    width_rho_ref_gev = 0.130
    g_ref = g_from_gamma(m_pi_ref_gev, m_rho_ref_gev, width_rho_ref_gev)
    L_gev = 4 / m_pi_ref_gev
    ref_list = [0.0737, 0.4702, 1.1333, 0.7509, 0.1123, 0.1452, 0.1335, 0.0192]
    a_n_sq = np.zeros(8)
    for n in range(8):
        a_n_sq[n], k_n, ff = get_amplitudes(
            n + 1,
            L_gev,
            m_pi_ref_gev,
            phase_shift=argFpi,
            absfpi=absFpi,
            phase_args=(m_pi_ref_gev, m_rho_ref_gev, g_ref),
            absfpi_args=(m_pi_ref_gev, m_rho_ref_gev, g_ref),
        )
        print(
            "n = ",
            n + 1,
            " |A_n|^2 = ",
            a_n_sq[n],
            " |A_n|^2 /3 m_pi^3 = ",
            a_n_sq[n] / (3 * m_pi_ref_gev**3),
        )
        print("n = ", n + 1, " k_n / mpi= ", k_n / m_pi_ref_gev, " F^2 = ", ff**2)
        assert abs(a_n_sq[n] / (3 * m_pi_ref_gev**3) - ref_list[n]) < 1e-3
    return


if __name__ == "__main__":
    test_amplitudes()
