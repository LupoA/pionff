import numpy as np
from pionff.params import (
    mass_pi0_GeV,
    mass_rho0_GeV,
    Lref_fm,
    gev_fm_conversion,
    mass_muon_GeV,
)
from pionff.mll.observables import solve_ll, mll_amu
from pionff.formfactors.gkpry import absFpi, argFpi, py_amu


def mll_py(L, m_muon, m_pi, m_rho, x0min, x0max=np.inf):
    print("GKPRY\n")

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
    x0_min = 0  # 1.628 * gev_fm_conversion

    yp = mll_py(
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pi0_GeV,
        mass_rho0_GeV,
        x0min=x0_min,
        x0max=np.inf,
    )
