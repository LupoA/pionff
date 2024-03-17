from pionff.formfactors.gounaris_sakurai import g_from_gamma, argFpi, absFpi
from pionff.mll.fv_amplitudes import get_amplitudes
from pionff.mll.fv_energies import get_energies
from pionff.mll.observables import solve_ll, mll_amu
import numpy as np


def test_fv_obs():
    m_pi_ref_gev = 0.13957
    m_rho_ref_gev = 0.773
    width_rho_ref_gev = 0.130
    g_ref = g_from_gamma(m_pi_ref_gev, m_rho_ref_gev, width_rho_ref_gev)
    L_gev = 4 / m_pi_ref_gev
    a_ref = np.zeros(8)
    e_ref = np.zeros(8)
    for n in range(8):
        a_ref[n], k_n, ff = get_amplitudes(
            n + 1,
            L_gev,
            m_pi_ref_gev,
            phase_shift=argFpi,
            absfpi=absFpi,
            phase_args=(m_pi_ref_gev, m_rho_ref_gev, g_ref),
            absfpi_args=(m_pi_ref_gev, m_rho_ref_gev, g_ref),
        )
        _, _, _, e_ref[n] = get_energies(
            L_gev, n + 1, m_pi_ref_gev, argFpi, m_pi_ref_gev, m_rho_ref_gev, g_ref
        )

    e_cmp, a_cmp = solve_ll(
        8,
        L_gev,
        m_pi_ref_gev,
        phase_shift=argFpi,
        absfpi=absFpi,
        phase_args=(m_pi_ref_gev, m_rho_ref_gev, g_ref),
        absfpi_args=(m_pi_ref_gev, m_rho_ref_gev, g_ref),
    )

    assert np.allclose(e_cmp, e_ref), "Failed testing function 'solve_ll'."
    assert np.allclose(a_cmp, a_ref), "Failed testing function 'solve_ll'."


def test_mllgs():
    from pionff.params import (
        mass_pi0_GeV,
        mass_rho0_GeV,
        g_ppr_0,
        Lref_fm,
        gev_fm_conversion,
        mass_muon_GeV,
    )

    L = Lref_fm * gev_fm_conversion
    m_pi = mass_pi0_GeV
    m_rho = mass_rho0_GeV
    m_muon = mass_muon_GeV
    g_ppr = g_ppr_0
    x0min = 0
    x0max = np.inf
    from pionff.formfactors.gounaris_sakurai import gs_amu

    amu_iv = gs_amu(m_muon, m_pi, m_rho, g_ppr, x0min=x0min, x0max=x0max)

    e_n, asq_n = solve_ll(
        8,
        L,
        m_pi,
        phase_shift=argFpi,
        absfpi=absFpi,
        phase_args=(m_pi, m_rho, g_ppr),
        absfpi_args=(m_pi, m_rho, g_ppr),
    )
    amu_fv, err_amu_fv = mll_amu(m_muon, e_n, asq_n, x0min=x0min, x0max=x0max)

    amu_iv *= 1e10
    amu_fv *= 1e10
    err_amu_fv *= 1e10

    assert (
        amu_iv - amu_fv - 18.806 < 1e-2
    ), "Failed checking finite volume effect of a_mu from MLLGS using reference value"
