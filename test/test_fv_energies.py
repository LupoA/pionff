# from pionff.params import mass_pi0_GeV, mass_rho0_GeV, width_rho0_neutral, g_ppr_0, Lref_fm, gev_fm_conversion
from pionff.formfactors.gounaris_sakurai import g_from_gamma, argFpi
from pionff.mll.fv_energies import get_energies


def test_energies():
    """
    test against table 1 of
    https://arxiv.org/pdf/1306.2532.pdf
    """
    m_pi_ref_gev = 0.13957
    m_rho_ref_gev = 0.773
    width_rho_ref_gev = 0.130
    g_ref = g_from_gamma(m_pi_ref_gev, m_rho_ref_gev, width_rho_ref_gev)
    L_gev = 4 / m_pi_ref_gev
    ref_list = [1.548, 2.133, 2.559, 2.831, 3.171, 3.581, 3.912, 4.459]
    for n in range(0, 8):
        q, k, zero_index, energy = get_energies(
            L_gev, n + 1, m_pi_ref_gev, argFpi, m_pi_ref_gev, m_rho_ref_gev, g_ref
        )
        assert (
            abs((k / m_pi_ref_gev) - ref_list[n]) < 1e-3
        ), "Failed testing FV energies against 1306.2532"
    return
