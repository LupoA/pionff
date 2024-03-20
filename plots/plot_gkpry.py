from pionff.formfactors.gkpry import phase_shift, absFpi
import numpy as np
import matplotlib.pyplot as plt
from pionff.params import mass_pi0_GeV, mass_rho0_GeV, g_ppr_0
from pionff.formfactors.gounaris_sakurai import (
    absFpi as gs_absfpi,
    argFpi as gs_phaseshift,
)


def plot_phase_shift():
    e_range = np.linspace(mass_pi0_GeV * 2, 1.6, 150)  # GeV
    plt.plot(
        e_range,
        57.2958 * phase_shift(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV),
        label=r"$\delta_{11}(E) \;$ (Pelaez, Yndurain)",
    )

    plt.plot(
        e_range,
        57.2958
        * gs_phaseshift(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0),
        label=r"$\delta_{11}(E) \;$ (Gounaris, Sakurai)",
    )

    plt.xlabel(r"$E \; [GeV] $")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid()
    plt.show()


def plot_abs_fpi():
    e_range = np.linspace(mass_pi0_GeV * 2, 2.2, 150)  # GeV
    plt.plot(
        e_range,
        absFpi(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV),
        label=r"$|F_\pi(E)| \;$ (Pelaez, Yndurain)",
    )

    plt.plot(
        e_range,
        gs_absfpi(e_range, m_pi=mass_pi0_GeV, m_rho=mass_rho0_GeV, g_ppr=g_ppr_0),
        label=r"$|F_\pi(E)| \;$ (Gounaris, Sakurai)",
    )

    plt.xlabel(r"$E \; [GeV] $")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_phase_shift()
    plot_abs_fpi()
