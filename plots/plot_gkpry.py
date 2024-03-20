from pionff.formfactors.gkpry import (
    phase_shift,
    absFpi,
    phase_shift_errors,
    par_UFD,
    par_CFD,
)
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


def plot_phase_shift_errors():
    e_range = np.linspace(2 * mass_pi0_GeV, 1.4, 300)

    results_mean_range_CFD = phase_shift(e_range, mass_pi0_GeV, mass_rho0_GeV, par_CFD)
    results_mean_range_UFD = phase_shift(e_range, mass_pi0_GeV, mass_rho0_GeV, par_UFD)
    results_std_range_CFD = phase_shift_errors(
        e_range, mass_pi0_GeV, mass_rho0_GeV, par_CFD, error="max"
    )
    results_std_range_UFD = phase_shift_errors(
        e_range, mass_pi0_GeV, mass_rho0_GeV, par_UFD, error="max"
    )

    # Create the plot for a range of e values
    plt.figure(figsize=(10, 6))
    plt.fill_between(
        e_range,
        57.2958 * (results_mean_range_CFD - results_std_range_CFD),
        57.2958 * (results_mean_range_CFD + results_std_range_CFD),
        label="GKPRY (CFD)",
        alpha=0.8,
    )

    plt.fill_between(
        e_range,
        57.2958 * (results_mean_range_UFD - results_std_range_UFD),
        57.2958 * (results_mean_range_UFD + results_std_range_UFD),
        label="GKPRY (UFD)",
        alpha=0.8,
    )

    plt.xlabel("E [GeV]")
    plt.ylabel(r"$\delta_1^1(E)$")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_phase_shift_errors()
    exit()
    plot_phase_shift()
    plot_abs_fpi()
