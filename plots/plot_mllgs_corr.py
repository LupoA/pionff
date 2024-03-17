import numpy as np
import matplotlib.pyplot as plt
from pionff.params import (
    mass_pi0_GeV,
    mass_rho0_GeV,
    g_ppr_0,
    Lref_fm,
    gev_fm_conversion,
)
from pionff.mll.observables import solve_ll, mll_corr
from pionff.formfactors.gounaris_sakurai import argFpi, absFpi
import logging


def plot_LL_series_convergence():
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_states = 8
    L_gev = Lref_fm * gev_fm_conversion
    times_fm = [0.05, 0.1, 0.2, 0.5, 0.8, 1, 1.5, 2]
    e_n, a_n = solve_ll(
        8,
        L_gev,
        mass_pi0_GeV,
        phase_shift=argFpi,
        absfpi=absFpi,
        phase_args=(mass_pi0_GeV, mass_rho0_GeV, g_ppr_0),
        absfpi_args=(mass_pi0_GeV, mass_rho0_GeV, g_ppr_0),
    )
    for i in range(len(times_fm)):
        terms = np.zeros(n_states)
        corr = 0
        for _n in range(n_states):
            corr += a_n[_n] * np.exp(-times_fm[i] * e_n[_n]) / 3
            terms[_n] = corr
        _, err = mll_corr(times_fm[i], e_n, a_n)
        plt.plot(
            np.linspace(1, n_states, n_states),
            terms,
            label="t = {:2f} [fm]".format(times_fm[i]),
            color=color_cycle[i],
            marker="o",
        )
        plt.axhline(corr, linestyle="--", color=color_cycle[i])
        logging.info("terms at time t = ", times_fm[i], terms)
        logging.info("sum at time t = ", times_fm[i], corr)
        logging.info("error at time t = ", times_fm[i], err)
    plt.legend(fontsize="large")
    plt.yscale("log")
    plt.title(r"$m_\pi L = {:2f}$".format(mass_pi0_GeV * L_gev), fontsize="x-large")
    plt.ylabel(r"$\sum_n^{n_{\rm max}} |A_n|^2 \; \exp(-t E_n)$", fontsize="x-large")
    plt.xlabel(r"$n_{\rm max}$", fontsize="x-large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.show()


if __name__ == "__main__":
    plot_LL_series_convergence()
