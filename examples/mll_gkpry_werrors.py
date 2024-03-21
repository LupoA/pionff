import numpy as np
from pionff.params import (
    mass_pi0_GeV,
    mass_rho0_GeV,
    Lref_fm,
    gev_fm_conversion,
    mass_muon_GeV,
)
from pionff.mll.observables import solve_ll, mll_amu
from pionff.formfactors.gkpry import (
    par_CFD,
    create_phase_instance,
    create_absFpi_instance,
)
from pionff.formfactors.infinite_volume import a_mu_from_rho_iv


def mll_py(L, n_copies, m_muon, m_pi, m_rho, x0min, x0max=np.inf):
    amu_list = []
    random_integers = np.random.randint(0, 1e5, size=n_copies)
    for c in range(n_copies):
        amu_iv = a_mu_from_rho_iv(
            m_muon,
            m_pi,
            create_absFpi_instance,
            m_pi,
            m_rho,
            par_CFD,
            "uniform",
            random_integers[c],
            x0min=x0min,
            x0max=x0max,
        )

        amu_list.append(amu_iv)

        e_n, asq_n = solve_ll(
            8,
            L,
            m_pi,
            phase_shift=create_phase_instance,
            absfpi=create_absFpi_instance,
            phase_args=(m_pi, m_rho, par_CFD, "uniform", random_integers[c]),
            absfpi_args=(m_pi, m_rho, par_CFD, "uniform", random_integers[c]),
        )

        amu_fv, err_amu_fv = mll_amu(m_muon, e_n, asq_n, x0min=x0min, x0max=x0max)

        amu_iv *= 1e10
        amu_fv *= 1e10
        err_amu_fv *= 1e10
        diff = amu_iv - amu_fv

        print("Iteratio ", n_copies, "finiteL diff = ", diff)

    std = np.std(diff)
    print("Error from parameters is ", std)
    mean = np.mean(diff)
    print("mean +- std", mean, std)
    return


if __name__ == "__main__":
    x0_min = 0  # 1.628 * gev_fm_conversion
    n_copies = 3
    mll_py(
        Lref_fm * gev_fm_conversion,
        n_copies,
        mass_muon_GeV,
        mass_pi0_GeV,
        mass_rho0_GeV,
        x0min=x0_min,
        x0max=np.inf,
    )
