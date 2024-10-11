import numpy as np
from pionff.params import (
    mass_pi0_GeV,
    Lref_fm,
    gev_fm_conversion,
    mass_muon_GeV,
)
from pionff.utils.debug_opt import timeit
from pionff.params import DEBUG_MODE
from pionff.mll.observables import solve_ll, mll_amu_window
from pionff.hp.fv_corrections_for_bmw import correction_amu_window_alpha
from pionff.formfactors.dhmz import (
    argFpi,
    absFpi_w_alpha,
    dhmz_amu_window,
    create_par_toys,
    par_KLOE_nominal,
)


@timeit(DEBUG_MODE)
def mll_DHMZ(L, m_muon, m_pi, par, x0min, x0_cut_dd, window):
    print("\n Starting MLL evaluation \n")
    amu_iv = dhmz_amu_window(
        m_muon,
        m_pi,
        par,
        x0min=x0min,
        x0_cut_dd=x0_cut_dd,
        window=window,
        include_alpha=True,
    )

    print("Window : ", window)
    print("Infinite L = ", amu_iv * 1e10)

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

    print("Ran MLL with:")
    print("\t L = ", L / gev_fm_conversion)
    print("\t m_pi = ", m_pi)
    print("\t parameters = ", par)
    print("\t m_muon = ", m_muon)
    print("\t m_pi L = ", m_pi * L)
    print("Results MLL: ")
    print("\t Infinite L = ", amu_iv)
    print("\t Finite L = ", amu_fv, " +/-", err_amu_fv)
    print("\t Infinite - Finite = ", amu_iv - amu_fv, "+/-", err_amu_fv)

    return (amu_iv - amu_fv), err_amu_fv, amu_iv, e_n, asq_n


@timeit(DEBUG_MODE)
def hp_DHMZ(window, x0_cut_dd, x0_mll_hp_match, L, m_muon, m_pi, par):
    """
    x0 min hardcoded to zero
    """
    print("\n Starting HP evaluation \n")
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

    print("HP Central result ", delta_amu * 1e10)
    print("HP Truncation error ", err * 1e10)

    return delta_amu * 1e10, err * 1e10


def run(par, window, full_output=False):
    """
    works at a single x0_cut
    """

    x0_dd_matching = 2.8
    x0_mll_hp_matching = 1.682
    mass_pion = mass_pi0_GeV

    result_from_hp, err_from_hp = hp_DHMZ(
        window,
        x0_cut_dd=x0_dd_matching * gev_fm_conversion,
        x0_mll_hp_match=x0_mll_hp_matching * gev_fm_conversion,
        L=Lref_fm * gev_fm_conversion,
        m_muon=mass_muon_GeV,
        m_pi=mass_pi0_GeV,
        par=par,
    )

    result_from_mll, error_from_mll, amu_iv, e_n, asq_n = mll_DHMZ(
        Lref_fm * gev_fm_conversion,
        mass_muon_GeV,
        mass_pion,
        par,
        x0min=x0_mll_hp_matching * gev_fm_conversion,
        x0_cut_dd=x0_dd_matching * gev_fm_conversion,
        window=window,
    )

    print("Results w $alpha_V$ \n")
    print("window : ", r"\text{", window, "}", r" \\ ")
    print("x0 cut data [fm] ", x0_dd_matching, r" \\ ")
    print("x0, cut hp-mll [fm]", x0_mll_hp_matching, r" \\ ")
    print("$m_\pi$ ", mass_pion, r" \\ ")
    print("$m_\pi$ L", mass_pion * Lref_fm * gev_fm_conversion, r" \\ ")
    print("DHMZ par ID: ", r"\text{", par["id"], "}", r" \\ ")
    print(
        "DHMZ par b0, b1, mrho, ",
        r"$\alpha_V$: ",
        par["b0"],
        par["b1"],
        par["mrho"],
        par["alpha_V"],
        r" \\ ",
    )
    print("$E_n / m_\pi$", e_n / mass_pion, r" \\ ")
    print("$|A_n|^2 / m^3_\pi$", asq_n / (mass_pion**3), r" \\ ")
    print("MLL contribution : ", result_from_mll, error_from_mll, r" \\ ")
    print("HP contribution : ", result_from_hp, err_from_hp, r" \\ ")
    print(
        "full fullerr ",
        result_from_mll + result_from_hp,
        err_from_hp + error_from_mll,
    )

    if not full_output:
        return result_from_mll + result_from_hp, err_from_hp + error_from_mll
    else:
        return result_from_mll, result_from_hp, error_from_mll, err_from_hp, amu_iv


def run_werrors(par, window, nsamples=100, seed=42):
    import json

    output_filename = (
        "output_FV_wAlphaV_using_"
        + str(par["id"])
        + "_seed_"
        + str(seed)
        + "_Nsamples_"
        + str(nsamples)
        + "_"
        + ".json"
    )
    print("Filename : ", output_filename)
    toys, cov, _ = create_par_toys(par, size=nsamples, seed=seed, return_cov=True)

    list_res = []
    list_res_hp = []
    list_res_mll = []
    list_ERR_hp = []
    list_ERR_mll = []
    list_IV_amu = []
    list_err = []

    with open(output_filename, "w") as json_file:  # Open the file once before the loop
        for n in range(len(toys[:, 0])):
            print("\n SAMPLE ", n, "\n")

            _par = {
                "b0": toys[n, 0],
                "b1": toys[n, 1],
                "mrho": toys[n, 2],
                "alpha_V": toys[n, 3],
                "id": par["id"],
            }

            result_from_mll, result_from_hp, error_from_mll, err_from_hp, amu_iv = run(
                _par, window, True
            )
            list_res.append(result_from_mll + result_from_hp)
            list_err.append(err_from_hp + error_from_mll)
            list_res_hp.append(result_from_hp)
            list_res_mll.append(result_from_mll)
            list_ERR_hp.append(err_from_hp)
            list_ERR_mll.append(error_from_mll)
            list_IV_amu.append(amu_iv)

            sample_data = {
                "sample_n": n,
                "b0": _par["b0"],
                "b1": _par["b1"],
                "mrho": _par["mrho"],
                "alphaV": _par["alpha_V"],
                "result": result_from_mll + result_from_hp,
                "error": err_from_hp + error_from_mll,
                "result_from_mll": result_from_mll,
                "error_from_mll": error_from_mll,
                "result_from_hp": result_from_hp,
                "error_from_hp": err_from_hp,
                "amu_iv_from_match_to_cut": amu_iv,
            }
            json.dump(sample_data, json_file, indent=4)

    # Calculate mean and std for each list
    mean_res = np.mean(list_res)
    std_res = np.std(list_res)
    mean_res_hp = np.mean(list_res_hp)
    std_res_hp = np.std(list_res_hp)
    mean_res_mll = np.mean(list_res_mll)
    std_res_mll = np.std(list_res_mll)
    mean_ERR_hp = np.mean(list_ERR_hp)
    std_ERR_hp = np.std(list_ERR_hp)
    mean_ERR_mll = np.mean(list_ERR_mll)
    std_ERR_mll = np.std(list_ERR_mll)
    mean_IV_amu = np.mean(list_IV_amu)
    std_IV_amu = np.std(list_IV_amu)
    mean_QUAD_err = np.mean(list_err)
    std_QUAD_err = np.std(list_err)

    # Print mean and std for each list
    print("Mean of list_res:", mean_res)
    print("Std of list_res:", std_res)
    print("Mean of list_res_hp:", mean_res_hp)
    print("Std of list_res_hp:", std_res_hp)
    print("Mean of list_res_mll:", mean_res_mll)
    print("Std of list_res_mll:", std_res_mll)
    print("Mean of list_ERR_hp:", mean_ERR_hp)
    print("Std of list_ERR_hp:", std_ERR_hp)
    print("Mean of list_ERR_mll:", mean_ERR_mll)
    print("Std of list_ERR_mll:", std_ERR_mll)
    print("Mean of list_IV_amu:", mean_IV_amu)
    print("Std of list_IV_amu:", std_IV_amu)
    print("Mean of list_QUAD_err:", mean_QUAD_err)
    print("Std of list_QUAD_err:", std_QUAD_err)

    results_export = {
        "avg_fv": mean_res,
        "std_fv": std_res,
        "avg_HP": mean_res_hp,
        "std_HP": std_res_hp,
        "avg_MLL": mean_res_mll,
        "std_MLL": std_res_mll,
        "avg_quad_truncation": mean_QUAD_err,
        "std_quad_truncation": std_QUAD_err,
        "mean_trunc_HP": mean_ERR_hp,
        "std_trunc_HP": std_ERR_hp,
        "mean_trunc_MLL": mean_ERR_mll,
        "std_trunc_MLL": std_ERR_mll,
        "mean_iv_match_to_cut": mean_IV_amu,
        "std_iv_match_to_cut": std_IV_amu,
    }

    with open(
        "AveragedResults_" + output_filename, "w"
    ) as f:  # Open file in append mode
        json.dump(results_export, f, indent=4)

    return 0


if __name__ == "__main__":
    parameter_set = par_KLOE_nominal
    # run_werrors(parameter_set, window='full', nsamples=50, seed=42)

    res_ID, err_ID = run(parameter_set, "04-10")
    exit()

    res_full, err_full = run(parameter_set, "00-28")
    print("\n\n")
    res_SD, err_SD = run(parameter_set, "00-04")
    print("\n\n")
    print("\n\n")
    res_LD, err_LD = run(parameter_set, "10-28")
    print("\n\n")
    print(
        "full - sum_windows :",
        res_full - (res_SD + res_ID + res_LD),
        "err on full :",
        err_full,
    )
    res_full, err_full = run(parameter_set, "28-35")
    exit()
    res_full, err_full = run(parameter_set, "15-19")
    exit()
