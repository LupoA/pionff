import numpy as np
from pionff.params import (
    mass_pi0_GeV,
    mass_rho0_GeV,
    width_rho0_neutral,
    g_ppr_0,
    mass_muon_GeV,
)
from pionff.formfactors.gounaris_sakurai import (
    g_from_gamma,
    argFpi,
    mSquare,
    gs_amu,
    absFpi,
)
from pionff.formfactors.infinite_volume import a_mu_iv, a_mu_from_rho_iv
from pionff.formfactors.omnes import omnes_function


def test_gs_functions():
    g0 = g_from_gamma(
        m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, gamma_rho=width_rho0_neutral
    )
    assert g0 - g_ppr_0, "Failed testing g_from_gamma."
    diff = 90 - (
        57.2958
        * argFpi(
            e=mass_rho0_GeV + 1e-8,
            m_pi=mass_pi0_GeV,
            m_rho=mass_rho0_GeV,
            g_ppr=g_ppr_0,
        )
    )
    assert diff < 1e-3, "Failed testing argFpi."
    diff = mass_rho0_GeV - np.sqrt(
        mSquare(e=mass_rho0_GeV, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0)
    )
    assert diff < 1e-8, "Failed testing mSquare(m_rho)."

    diff = 1 - omnes_function(
        0, 4 * mass_pi0_GeV**2, argFpi, mass_pi0_GeV, mass_rho0_GeV, g_ppr_0
    )
    assert diff < 1e-8, "Failed testing omnes(s=0)."

    amu_iv = gs_amu(
        mass_muon_GeV, mass_pi0_GeV, mass_rho0_GeV, g_ppr_0, x0min=0, x0max=np.inf
    )
    assert (
        amu_iv - 445.876 < 1e-2
    ), "Failed a_mu in infinite volume against known value."

    a_mu_from_generic_function = a_mu_iv(
        mass_muon_GeV,
        mass_pi0_GeV,
        absFpi,
        mass_pi0_GeV,
        mass_rho0_GeV,
        g_ppr_0,
        x0min=0,
        x0max=np.inf,
    )

    assert (
        a_mu_from_generic_function - amu_iv < 1e-8
    ), "Failed comparing a_mu in infinite volume from a generic function specialised in GS and a GS-hardcoded function."

    a_mu_from_rho = a_mu_from_rho_iv(
        mass_muon_GeV,
        mass_pi0_GeV,
        absFpi,
        mass_pi0_GeV,
        mass_rho0_GeV,
        g_ppr_0,
        x0min=0,
        x0max=np.inf,
    )
    print("diff", abs(a_mu_from_rho - amu_iv) * 1e10)
    assert abs(a_mu_from_rho - amu_iv) * 1e10 < 1e-2


if __name__ == "__main__":
    test_gs_functions()
