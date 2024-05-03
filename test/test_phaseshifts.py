from pionff.formfactors.gkpry import phase_shift as gkpry_phase
from pionff.formfactors.gkpry import par_CFD
from pionff.formfactors.gounaris_sakurai import argFpi as gs_phase
from pionff.params import mass_pi0_GeV, mass_rho0_GeV, g_ppr_0


def test_phases():
    delta_at_rho = 57.2958 * gkpry_phase(
        mass_rho0_GeV - 1e-6,
        m_pi=mass_pi0_GeV,
        m_rho=mass_rho0_GeV,
        par=par_CFD,
    )
    assert delta_at_rho - 90 < 1e-4, "Failed testing delta(M_rho) for GKPRY."

    delta_at_rho = 57.2958 * gs_phase(
        mass_rho0_GeV - 1e-6, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0
    )
    assert delta_at_rho - 90 < 1e-4, "Failed testing delta(M_rho) for GS."

    assert (
        gkpry_phase(
            2 * mass_pi0_GeV, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, par=par_CFD
        )
        == 0
    ), "Phase shift does not vanish at threshold for GKPRY."

    assert (
        gs_phase(
            2 * mass_pi0_GeV, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0
        )
        == 0
    ), "Phase shift does not vanish at threshold for GS."


test_phases()
