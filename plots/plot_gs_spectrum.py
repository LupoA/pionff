import numpy as np
import matplotlib.pyplot as plt
from pionff.params import mass_pi0_GeV, mass_rho0_GeV, g_ppr_0
from pionff.mll.observables import solve_ll, mll_ssd
from pionff.formfactors.gounaris_sakurai import argFpi, absFpi, spectral_density
from scipy.integrate import quad_vec
from scipy.stats import norm


def plot_infinite_volume_density():
    energies_gev = np.linspace(0, 1, 100)
    plt.plot(
        energies_gev,
        spectral_density(
            energies_gev, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0
        ),
        label=r"$\rho_{L=\infty}(E)$",
    )
    plt.legend(fontsize="large")
    plt.title(r"Using Gounaris Sakurai")
    # plt.ylabel(r'', fontsize='x-large')
    plt.xlabel(r"$E$ [GeV]", fontsize="x-large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.show()
    return


def smeared_gs(e, sigma):
    def integrand(x, e, sigma):
        return norm.pdf(e, loc=x, scale=sigma) * spectral_density(
            x, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0
        )

    a = 2 * mass_pi0_GeV + 1e-6
    b = 50
    res, _ = quad_vec(integrand, a, b, args=(e, sigma))
    return res / (e * e)


def plot_finite_volume_smeared_dennsity():
    energies_gev = np.linspace(2 * mass_pi0_GeV + 1e-6, 1.15, 200)
    n_states = 8
    mL_list_gev = [3, 4, 6.3]
    sigma = mass_pi0_GeV / 12
    for _mL in mL_list_gev:
        L = _mL / mass_pi0_GeV
        e_n, a_n = solve_ll(
            n_states,
            L,
            mass_pi0_GeV,
            phase_shift=argFpi,
            absfpi=absFpi,
            phase_args=(mass_pi0_GeV, mass_rho0_GeV, g_ppr_0),
            absfpi_args=(mass_pi0_GeV, mass_rho0_GeV, g_ppr_0),
        )
        plt.plot(
            energies_gev,
            mll_ssd(energies_gev, sigma, e_n, a_n),
            label=r"$\rho_{L=}(E) " + r" \; m_\pi L={:2.2f}$".format(_mL),
            ls="-",
        )

    plt.plot(
        energies_gev,
        smeared_gs(energies_gev, sigma),
        label=r"$L = \infty$",
        ls="--",
        color="k",
    )

    plt.title(r"Using Gounaris Sakurai")
    plt.legend(fontsize="large")
    plt.xlabel(r"$E$ [GeV]", fontsize="x-large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.show()
    return


if __name__ == "__main__":
    plot_finite_volume_smeared_dennsity()
