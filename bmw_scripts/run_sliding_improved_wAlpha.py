import numpy as np
from pionff.params import (
    mass_pi0_GeV,
    Lref_fm,
    gev_fm_conversion,
    mass_muon_GeV,
)
from pionff.hp.regularpart import amu_reg
from pionff.mll.observables import solve_ll, mll_amu_window
from pionff.hp.fv_corrections_for_bmw import correction_amu_window_alpha
from pionff.formfactors.dhmz import (
    argFpi,
    absFpi,
    absFpi_w_alpha,
    dhmz_amu_window,
    par_BaBar_nominal,
    par_KLOE_nominal,
    par_CMD3_nominal,
)


def run_MLL_tail(L, m_muon, m_pi, x0min, window, x0_cut_dd):
    """
    silly game: in order to approximate the correlator,
    take the free discretised energies and, for the amplitudes,
    sample the infinite-volume spectral density x E^2
    """
    from pionff.formfactors.gounaris_sakurai import absFpi as GS_fpi
    from pionff.params import mass_rho0_GeV, g_ppr_0
    from pionff.mll.non_interacting_pions import free_fv_energies_and_amplitudes

    m_rho = mass_rho0_GeV
    g_ppr = g_ppr_0
    QMAX = 32
    CUT = 7
    e_n_list, a_n_list = free_fv_energies_and_amplitudes(m_pi, L, qmax=QMAX)
    e_n_list = e_n_list[CUT:]
    a_n_list = a_n_list[CUT:]
    e_n_array = np.array(e_n_list)
    a_n_array = np.array(a_n_list)

    n_states = 42
    w_n = np.zeros(n_states)
    a_n_array = a_n_array[:n_states]
    e_n_array = e_n_array[:n_states]
    for n in range(n_states):
        w_n[n] = a_n_array[n] * (GS_fpi(e_n_array[n], m_pi, m_rho, g_ppr) ** 2)
        # print(e_n_array[n], w_n[n])

    amu, err_amu = mll_amu_window(
        m_muon, e_n_array, w_n, window=window, x0min=x0min, x_cut_dd=x0_cut_dd
    )
    amu *= 1e10
    err_amu *= 1e10
    print("result mll tail : ", amu, " +- ", err_amu)
    return amu, err_amu


def mll_DHMZ(L, m_muon, m_pi, par, x0min, x0_cut_dd, window, w_alpha=True):
    tail_mll, _ = run_MLL_tail(
        L, m_muon, m_pi, x0min=x0min, window=window, x0_cut_dd=x0_cut_dd
    )

    amu_iv = dhmz_amu_window(
        m_muon,
        m_pi,
        par,
        x0min=x0min,
        x0_cut_dd=x0_cut_dd,
        window=window,
        include_alpha=w_alpha,
    )

    print("\tWindow : ", window)
    print("\tInfinite L = ", amu_iv * 1e10)

    if not w_alpha:
        e_n, asq_n = solve_ll(
            8,
            L,
            m_pi,
            phase_shift=argFpi,
            absfpi=absFpi,
            phase_args=(m_pi, par),
            absfpi_args=(m_pi, par),
        )
    else:
        e_n, asq_n = solve_ll(
            8,
            L,
            m_pi,
            phase_shift=argFpi,
            absfpi=absFpi_w_alpha,
            phase_args=(m_pi, par),
            absfpi_args=(m_pi, par),
        )

    print("E_n =", e_n)
    print("|A_n|^2 = ", asq_n)

    amu_fv, err_amu_fv = mll_amu_window(
        m_muon, e_n, asq_n, window=window, x0min=x0min, x_cut_dd=x0_cut_dd
    )

    amu_iv *= 1e10
    amu_fv *= 1e10
    err_amu_fv *= 1e10

    print("\tRan MLL with:")
    print("\t\t L = ", L / gev_fm_conversion)
    print("\t\t m_pi = ", m_pi)
    print("\t\t parameters = ", par)
    print("\t\t m_muon = ", m_muon)
    print("\t\t m_pi L = ", m_pi * L)
    print("Results MLL: ")
    print("\t\t Infinite L = ", amu_iv)
    print("\t\t Finite L = ", amu_fv, " +/-", err_amu_fv)
    print("\t\t Infinite - Finite = ", amu_iv - amu_fv, "+/-", err_amu_fv)

    return (amu_iv - amu_fv), err_amu_fv, amu_iv, e_n, asq_n, tail_mll


def hp_DHMZ(window, x0_cut_dd, x0_mll_hp_match, L, m_muon, m_pi, par):
    """
    x0 min hardcoded to zero
    """
    # correction_amu_window(window, x0_cut_dd, x0_cut_mll_hp, L, m_muon, m_pi, phase_shift, *args)
    delta_amu, err = correction_amu_window_alpha(
        window,
        x0_cut_dd,
        x0_mll_hp_match,
        L,
        m_muon,
        m_pi,
        par["alpha_V"],
        argFpi,
        m_pi,
        par,
    )

    reg_part = amu_reg(
        L=L,
        m_muon=m_muon,
        m_pi=m_pi,
        x0_min=1e-6,
        x0_max=x0_mll_hp_match,
        window=window,
        alpha=1 / 137,
    )

    reg_part *= 1e10
    print("t reg ", reg_part)
    delta_amu *= 1e10
    err *= 1e10
    delta_amu += reg_part

    return delta_amu, err, reg_part


def run(par, window, mllhp_cut):
    """
    works at a single x0_cut
    """

    x0_dd_matching = 2.8
    x0_mll_hp_matchinh = mllhp_cut
    mass_pion = mass_pi0_GeV

    result_from_mll, error_from_mll, amu_iv, e_n, asq_n, tail_mll = mll_DHMZ(
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pion,
        par,
        x0min=x0_mll_hp_matchinh * gev_fm_conversion,
        x0_cut_dd=x0_dd_matching * gev_fm_conversion,
        window=window,
        w_alpha=True,
    )

    result_from_hp, err_from_hp, reg_part = hp_DHMZ(
        window,
        x0_cut_dd=x0_dd_matching * gev_fm_conversion,
        x0_mll_hp_match=x0_mll_hp_matchinh * gev_fm_conversion,
        L=Lref_fm * gev_fm_conversion,
        m_muon=mass_muon_GeV,
        m_pi=mass_pi0_GeV,
        par=par,
    )

    print("Results\n")
    print("--")
    print("window : ", r"\text{", window, "}", r" \\ ")
    print("x0, cut hp-mll [fm]", x0_mll_hp_matchinh, r" \\ ")
    print("$m_\pi$ ", mass_pion, r" \\ ")
    print("$m_\pi$ L", mass_pion * Lref_fm * gev_fm_conversion, r" \\ ")
    print("DHMZ par ID: ", r"\text{", par["id"], "}", r" \\ ")
    print("DHMZ par b0, b1, mrho: ", par["b0"], par["b1"], par["mrho"], r" \\ ")
    print("$E_n / m_\pi$", e_n / mass_pion, r" \\ ")
    print("$|A_n|^2 / m^3_\pi$", asq_n / (mass_pion**3), r" \\ ")
    print(
        "MLL contribution (value, err, tail) : ",
        result_from_mll,
        error_from_mll,
        tail_mll,
        r" \\ ",
    )
    print("HP contribution : ", result_from_hp, err_from_hp, r" \\ ")
    print(
        "full fullerr ",
        result_from_mll + result_from_hp,
        err_from_hp + error_from_mll,
    )
    print(
        "full fullerr w free pions tail ",
        result_from_mll + result_from_hp,
        err_from_hp + error_from_mll,
    )
    print("--")

    return (
        result_from_mll,
        result_from_hp,
        error_from_mll,
        err_from_hp,
        tail_mll,
        reg_part,
    )


def slide_cut(par, window):
    # x0l = [1.0, 1.1, 1.2, 1.3, 1.682, 3]
    x0_list = np.array([1, 1.1, 1.182, 1.2, 1.3, 2.182])

    # x0_list = np.linspace(1.682, 1.682, 1)

    res = []
    err = []
    res_shifted = []
    res_shifted_tail = []
    res_MLL = []
    err_wtail = []
    err_MLL = []
    res_HP = []
    err_HP = []
    reg_part = []

    for i in range(len(x0_list)):
        print("x0: ", x0_list[i])
        (
            result_from_mll,
            result_from_hp,
            err_from_mll,
            err_from_hp,
            tail_mml,
            hp_reg_part,
        ) = run(par, window, x0_list[i])
        res.append(result_from_hp + result_from_mll)
        res_shifted.append(result_from_hp + result_from_mll - err_from_mll)
        res_shifted_tail.append(result_from_hp + result_from_mll - tail_mml)
        err.append(err_from_mll + err_from_hp)
        err_wtail.append(tail_mml + err_from_hp)
        res_MLL.append(result_from_mll)
        err_MLL.append(err_from_mll)
        res_HP.append(result_from_hp)
        err_HP.append(err_from_hp)
        reg_part.append(hp_reg_part)

    print(
        "x0 & res & err & res-shift & res-shift-tail & err-w-tail & res-MLL & err-MLL & res-HP & err-HP & hp-reg-part"
    )
    for i in range(len(x0_list)):
        print(
            x0_list[i],
            res[i],
            err[i],
            res_shifted[i],
            res_shifted_tail[i],
            err_wtail[i],
            res_MLL[i],
            err_MLL[i],
            res_HP[i],
            err_HP[i],
            reg_part[i],
        )

    """
    plt.errorbar(x0_list, res, err, label='sum', marker='o')
    plt.errorbar(x0_list, res_shifted, err, label='sum shifted', marker='s')
    plt.errorbar(x0_list, res_MLL, err_MLL, label='MLL', marker='v')
    plt.errorbar(x0_list, res_HP, err_HP, label='HP', marker='^')
    plt.xlabel(r"$x_0^{\rm switch} \; $[fm]")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid()
    plt.show()

    plt.errorbar(x0_list, res, err, label='sum w free pion tail', marker='o')
    plt.errorbar(x0_list, res_shifted_tail, err_wtail, label='sum shifted w free pion tail', marker='s')
    plt.errorbar(x0_list, res_MLL, err_MLL, label='MLL', marker='v')
    plt.errorbar(x0_list, res_HP, err_HP, label='HP', marker='^')
    plt.xlabel(r"$x_0^{\rm switch} \; $[fm]")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid()
    plt.show()"""

    return


if __name__ == "__main__":
    slide_cut(par_CMD3_nominal, "00-28")

    slide_cut(par_BaBar_nominal, "00-28")
    slide_cut(par_KLOE_nominal, "00-28")
    slide_cut(par_CMD3_nominal, "00-28")

    exit()
    slide_cut(par_BaBar_nominal, "04-10")
    slide_cut(par_KLOE_nominal, "04-10")
    slide_cut(par_CMD3_nominal, "04-10")

    slide_cut(par_BaBar_nominal, "00-04")
    slide_cut(par_KLOE_nominal, "00-04")
    slide_cut(par_CMD3_nominal, "00-04")

    slide_cut(par_BaBar_nominal, "15-19")
    slide_cut(par_KLOE_nominal, "15-19")
    slide_cut(par_CMD3_nominal, "15-19")

    slide_cut(par_BaBar_nominal, "28-35")
    slide_cut(par_KLOE_nominal, "28-35")
    slide_cut(par_CMD3_nominal, "28-35")

    slide_cut(par_BaBar_nominal, "10-28")
    slide_cut(par_KLOE_nominal, "10-28")
    slide_cut(par_CMD3_nominal, "10-28")
    exit()

    slide_cut(par_BaBar_nominal, "00-28")
    slide_cut(par_KLOE_nominal, "00-28")
    slide_cut(par_CMD3_nominal, "00-28")
    exit()

    slide_cut(par_BaBar_nominal, "10-28")
    slide_cut(par_CMD3_nominal, "10-28")
    exit()

    slide_cut(par_KLOE_nominal, "15-19")
    slide_cut(par_BaBar_nominal, "15-19")
    slide_cut(par_CMD3_nominal, "15-19")
    exit()

    slide_cut(par_BaBar_nominal, "04-10")
    slide_cut(par_KLOE_nominal, "04-10")
    slide_cut(par_CMD3_nominal, "04-10")
    exit()

    slide_cut(par_KLOE_nominal, "10-28")
    slide_cut(par_BaBar_nominal, "10-28")
    slide_cut(par_CMD3_nominal, "10-28")
    exit()

    slide_cut(par_BaBar_nominal, "28-35")
    slide_cut(par_KLOE_nominal, "28-35")
    slide_cut(par_CMD3_nominal, "28-35")
    exit()

    slide_cut(par_KLOE_nominal, "00-04")

    slide_cut(par_BaBar_nominal, "10-28")
    slide_cut(par_KLOE_nominal, "10-28")
    slide_cut(par_CMD3_nominal, "10-28")
    exit()
