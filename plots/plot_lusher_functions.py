import matplotlib.pyplot as plt
from pionff.utils.read_luscher_table import read_luscher_function
from pionff.mll.fv_energies import pi_n_minus_delta, find_zero, zero_phase_shift
from pionff.formfactors.gounaris_sakurai import argFpi
import numpy as np
from pionff.params import (
    gev_fm_conversion,
    Lref_fm,
    mass_pi0_GeV,
    mass_rho0_GeV,
    g_ppr_0,
)


def plot_table():
    qsq, phi_q_normalised, tan_phi, zeta00 = read_luscher_function()

    plt.plot(qsq, phi_q_normalised, label=r"$\phi(q)/(\pi q^2)$")
    plt.xlabel(r"$q^2$")
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(qsq, tan_phi, label=r"$\tan (\phi(q)) $")
    plt.xlabel(r"$q^2$")
    plt.ylim(top=1e2, bottom=-1e2)
    plt.xlim(left=0, right=1e1)
    plt.legend()
    plt.show()
    plt.close()

    qsq_filtered = []
    zeta00_filtered = []
    for q, z in zip(qsq, zeta00):
        if q <= 10:  # if q <= 40 and q.is_integer():
            qsq_filtered.append(q)
            zeta00_filtered.append(z)
    plt.plot(qsq_filtered, zeta00_filtered, label=r"$Z_{00}/(1; q^2)$")
    plt.xlabel(r"$q^2$")
    plt.ylim(top=1e2, bottom=-1e2)
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid()
    plt.show()


def plot_intersections():
    def flattened_pi_n_minus_delta(n):
        return pi_n_minus_delta(
            q, n, Lref_gev, mass_pi0_GeV, argFpi, mass_pi0_GeV, mass_rho0_GeV, g_ppr_0
        )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    qsq, phi_q_normalised, tan_phi, zeta00 = read_luscher_function()
    q = np.sqrt(qsq)
    Lref_gev = Lref_fm * gev_fm_conversion

    label_gs = r"$\pi - \delta_{\rm GS}(2\pi q/L)$"
    indx_array = np.zeros(8, dtype=int)
    indx_free = np.zeros(8, dtype=int)

    for n in range(0, 8):
        delta_array = flattened_pi_n_minus_delta(n + 1)
        plt.plot(q, delta_array, label="{:d}".format(n + 1) + label_gs)
        _, _, indx_array[n] = find_zero(
            qsq,
            phi_q_normalised,
            n + 1,
            Lref_gev,
            mass_pi0_GeV,
            argFpi,
            mass_pi0_GeV,
            mass_rho0_GeV,
            g_ppr_0,
        )
        _, _, indx_free[n] = find_zero(
            qsq,
            phi_q_normalised,
            n + 1,
            Lref_gev,
            mass_pi0_GeV,
            phase_shift=zero_phase_shift,
        )
        plt.scatter(q[indx_array[n]], delta_array[indx_array[n]], marker="*")
        plt.axvline(x=q[indx_free[n]], linestyle="--", color=color_cycle[n])
    plt.plot([], [], color="k", ls="", label="Vertical: non-interacting")
    plt.plot(q, phi_q_normalised * np.pi * qsq, label=r"$\phi(q)$", color="k", ls="--")
    plt.title(r"notice that $\sqrt{7}$ is not among the vertical lines")

    plt.xlabel(r"$q$ ")
    plt.xlim(left=0, right=4)
    plt.ylim(top=30)
    plt.legend(loc="lower center", ncol=4)
    plt.show()


if __name__ == "__main__":
    # plot_table()
    plot_intersections()
