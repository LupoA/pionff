import numpy as np
from pionff.params import (
    mass_pi0_GeV,
    mass_rho0_GeV,
    Lref_fm,
    gev_fm_conversion,
    mass_muon_GeV,
)
from pionff.mll.observables import solve_ll, mll_amu
from pionff.formfactors.gkpry import absFpi, argFpi, py_amu, par_CFD
from pionff.hp.fv_corrections import correction_amu
from pionff.formfactors.gkpry import phase_shift as gkpry_delta
import matplotlib.pyplot as plt
import json


def hp_py(L, m_muon, m_pi, m_rho, par, x0min, x0max):
    assert x0min == 0
    delta_amu, err = correction_amu(
        x0max,
        L,
        m_muon,
        m_pi,
        gkpry_delta,
        m_pi,
        m_rho,
        par,
    )

    print("Central result ", delta_amu * 1e10)
    print("Truncation error ", err * 1e10)

    return delta_amu * 1e10, err * 1e10


def mll_py(L, m_muon, m_pi, m_rho, x0min, x0max=np.inf):
    print("GKPRY\n")

    amu_iv = py_amu(m_muon, m_pi, m_rho, par_CFD, x0min=x0min, x0max=x0max)

    print("Infinite L = ", amu_iv * 1e10)

    e_n, asq_n = solve_ll(
        8,
        L,
        m_pi,
        phase_shift=argFpi,
        absfpi=absFpi,
        phase_args=(m_pi, m_rho, par_CFD),
        absfpi_args=(m_pi, m_rho, par_CFD),
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

    return (amu_iv - amu_fv), err_amu_fv


def slide_x0min():
    """
    combines mll and hp matching at different values of x0cut. No parameter error
    """
    x0_min_set = np.linspace(0.0, 4.25, 16)
    x0_min_set *= gev_fm_conversion

    res_mll = []
    trunc_mll = []
    relative_mll = []
    res_hp = []
    trunc_hp = []
    relative_hp = []

    sum_values = []
    errsum_values = []
    relative_sum = []

    for _x0min in x0_min_set:
        yp, err = mll_py(
            Lref_fm * gev_fm_conversion,
            mass_muon_GeV,
            mass_pi0_GeV,
            mass_rho0_GeV,
            x0min=_x0min,
            x0max=np.inf,
        )
        res_mll.append(yp)
        trunc_mll.append(err)
        relative_mll.append(err / yp)
        print("res mll : ", _x0min, yp, err)

        if _x0min == 0:
            hp = 0
            hperr = 0
            print("res hp : ", _x0min, hp, hperr)
            res_hp.append(hp)
            trunc_hp.append(hperr)
            relative_hp.append(0)
            relative_sum.append((err / yp))
        else:
            hp, hperr = hp_py(
                Lref_fm * gev_fm_conversion,
                mass_muon_GeV,
                mass_pi0_GeV,
                mass_rho0_GeV,
                par_CFD,
                0,
                _x0min,
            )
            print("res hp : ", _x0min, hp, hperr)
            res_hp.append(hp)
            trunc_hp.append(hperr)
            relative_hp.append(hperr / hp)
            relative_sum.append((hperr / hp) + (err / yp))

        sum_values.append(hp + yp)
        errsum_values.append(hperr + err)

    data = {
        "x0_min_set": x0_min_set.tolist(),
        "res_mll": res_mll,
        "trunc_mll": trunc_mll,
        "relative_mll": relative_mll,
        "res_hp": res_hp,
        "trunc_hp": trunc_hp,
        "relative_hp": relative_hp,
        "sum_values": sum_values,
        "errsum_values": errsum_values,
        "relative_sum": relative_sum,
    }

    with open("results_gkpry_inflated/sliding_x0cut_data.json", "w") as json_file:
        json.dump(data, json_file)

    x0_min_set /= gev_fm_conversion
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

    ax1.errorbar(
        x0_min_set,
        res_mll,
        trunc_mll,
        label=r"MLL from $[x_0^{\rm cut}, \infty)$ w\ truncation errors",
        marker="o",
    )
    ax1.errorbar(
        x0_min_set,
        res_hp,
        trunc_hp,
        label=r"HP from $[0, x_0^{\rm cut}]$ w\ truncation errors",
        marker="v",
    )
    ax1.errorbar(x0_min_set, sum_values, errsum_values, label="Sum", marker="^")
    ax1.fill_between(
        x0_min_set,
        17.30 - 0.52,
        17.30 + 0.52,
        color="gray",
        alpha=0.4,
        label=r"$x_0^{\rm cut} = (m_\pi L / 4)^2 / m_\pi$",
    )
    ax1.set_ylabel(r"$\delta_L a_\mu $", fontsize=14)
    ax1.tick_params(axis="both", which="both", labelsize=12)
    ax1.legend(fontsize=12)
    ax1.grid()

    ax2.plot(x0_min_set, relative_mll, label="MLL", marker="o")
    ax2.plot(x0_min_set, relative_hp, label="HP", marker="v")
    ax2.plot(x0_min_set, relative_sum, label="Sum", marker="^")
    ax2.set_xlabel(r"$x_0^{\rm cut} \; $ [fm]", fontsize=14)
    ax2.set_ylabel(r"$\rm{trunc} \;\;  \rm{err} / \delta_L a_\mu$", fontsize=14)
    ax2.tick_params(axis="both", which="both", labelsize=12)
    ax2.legend(fontsize=12)
    ax2.grid()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def run():
    """
    works at a single x0_cut
    """
    x0_cut = 1.682
    x_max = np.inf
    x_min = 0

    result_from_mll, error_from_mll = mll_py(
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pi0_GeV,
        mass_rho0_GeV,
        x0min=x0_cut * gev_fm_conversion,
        x0max=x_max * gev_fm_conversion,
    )

    result_from_hp, error_from_hp = hp_py(
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pi0_GeV,
        mass_rho0_GeV,
        par_CFD,
        x0min=x_min * gev_fm_conversion,
        x0max=x0_cut * gev_fm_conversion,
    )

    print("Results\n")
    print("x0 cut [fm] : ", x0_cut)
    print("MLL contribution : ", result_from_mll, error_from_mll)
    print("HP contribution : ", result_from_hp, error_from_hp)
    print(
        "sum / lin err / quad : ",
        result_from_mll + result_from_hp,
        error_from_hp + error_from_mll,
        np.sqrt(error_from_mll**2 + error_from_hp**2),
    )

    return


if __name__ == "__main__":
    run()
    exit()
    slide_x0min()
