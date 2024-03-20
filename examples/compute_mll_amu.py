import numpy as np
from pionff.params import (
    mass_pi0_GeV,
    mass_rho0_GeV,
    g_ppr_0,
    Lref_fm,
    gev_fm_conversion,
    mass_muon_GeV,
)
from pionff.mll.observables import solve_ll, mll_amu


def mll_gs(L, m_muon, m_pi, m_rho, g_ppr, x0min, x0max=np.inf):
    from pionff.formfactors.gounaris_sakurai import argFpi, absFpi
    from pionff.formfactors.gounaris_sakurai import gs_amu

    print("GOUNARI SAKURAI\n")
    amu_iv = gs_amu(m_muon, m_pi, m_rho, g_ppr, x0min=x0min, x0max=x0max)
    print("Infinite L = ", amu_iv * 1e10)
    e_n, asq_n = solve_ll(
        8,
        L,
        m_pi,
        phase_shift=argFpi,
        absfpi=absFpi,
        phase_args=(m_pi, m_rho, g_ppr),
        absfpi_args=(m_pi, m_rho, g_ppr),
    )
    print("E_n =", e_n)
    print("|A_n|^2 = ", asq_n)

    amu_fv, err_amu_fv = mll_amu(m_muon, e_n, asq_n, x0min=x0min, x0max=x0max)

    amu_iv *= 1e10
    amu_fv *= 1e10
    err_amu_fv *= 1e10

    print("Parameters:")
    print("Integration range : ", "[", x0min, ", ", x0max, "]")
    print("L = ", L * gev_fm_conversion)
    print("m_pi = ", m_pi)
    print("m_rho = ", m_rho)
    print("m_muon = ", m_muon)
    print("m_pi L = ", m_pi * L)
    print("Results : ")
    print("Infinite L = ", amu_iv)
    print("Finite L = ", amu_fv, " +/-", err_amu_fv)
    print("Infinite - Finite = ", amu_iv - amu_fv, "+/-", err_amu_fv)

    return amu_iv - amu_fv


def mll_py(L, m_muon, m_pi, m_rho, x0min, x0max=np.inf):
    from pionff.formfactors.gkpry import py_amu
    from pionff.formfactors.gkpry import absFpi, argFpi

    print("PELAEZ YNDURAIN\n")

    amu_iv = py_amu(m_muon, m_pi, m_rho, x0min=x0min, x0max=x0max)
    print("Infinite L = ", amu_iv * 1e10)

    e_n, asq_n = solve_ll(
        8,
        L,
        m_pi,
        phase_shift=argFpi,
        absfpi=absFpi,
        phase_args=(m_pi, m_rho),
        absfpi_args=(m_pi, m_rho),
    )
    print("E_n =", e_n)
    print("|A_n|^2 = ", asq_n)

    amu_fv, err_amu_fv = mll_amu(m_muon, e_n, asq_n, x0min=x0min, x0max=x0max)

    amu_iv *= 1e10
    amu_fv *= 1e10
    err_amu_fv *= 1e10

    print("Parameters:")
    print("Integration range : ", "[", x0min, ", ", x0max, "]")
    print("L = ", L * gev_fm_conversion)
    print("m_pi = ", m_pi)
    print("m_rho = ", m_rho)
    print("m_muon = ", m_muon)
    print("m_pi L = ", m_pi * L)
    print("Results : ")
    print("Infinite L = ", amu_iv)
    print("Finite L = ", amu_fv, " +/-", err_amu_fv)
    print("Infinite - Finite = ", amu_iv - amu_fv, "+/-", err_amu_fv)

    return amu_iv - amu_fv


if __name__ == "__main__":
    x0_min = 1.628 * gev_fm_conversion

    yp = mll_py(
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pi0_GeV,
        mass_rho0_GeV,
        x0min=x0_min,
        x0max=np.inf,
    )

    gs = mll_gs(
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pi0_GeV,
        mass_rho0_GeV,
        g_ppr_0,
        x0min=x0_min,
        x0max=np.inf,
    )

    print("gs - py = ", abs(gs - yp))
