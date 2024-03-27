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
    par_CFD_2sigma,
    create_phase_instance,
    create_absFpi_instance,
    py_amu,
    argFpi,
    absFpi,
)
from pionff.formfactors.infinite_volume import a_mu_from_rho_iv
import json


def mll_central(L, m_muon, m_pi, m_rho, par, x0min, x0max):
    print("Computing central values")
    amu_iv = py_amu(m_muon, m_pi, m_rho, par=par, x0min=x0min, x0max=x0max)

    print("Infinite L = ", amu_iv * 1e10)

    e_n, asq_n = solve_ll(
        8,
        L,
        m_pi,
        phase_shift=argFpi,
        absfpi=absFpi,
        phase_args=(m_pi, m_rho, par),
        absfpi_args=(m_pi, m_rho, par),
    )

    print("E_n =", e_n)
    print("|A_n|^2 = ", asq_n)

    amu_fv, err_amu_fv = mll_amu(m_muon, e_n, asq_n, x0min=x0min, x0max=x0max)

    amu_iv *= 1e10
    amu_fv *= 1e10
    err_amu_fv *= 1e10

    return (amu_iv - amu_fv), amu_iv, amu_fv, err_amu_fv


def mll_py_err(file_path, L, n_copies, m_muon, m_pi, m_rho, par, x0min, x0max):
    amu_iv_list = []
    amu_fv_list = []
    amu_diff_list = []
    amu_err_list = []

    random_integers = np.random.randint(0, 1e5, size=n_copies)

    for c in range(n_copies):
        print("Iteraciton ", c, "using seed ", random_integers[c])
        amu_iv = a_mu_from_rho_iv(
            m_muon,
            m_pi,
            create_absFpi_instance,
            m_pi,
            m_rho,
            par,
            "uniform",
            random_integers[c],
            x0min=x0min,
            x0max=x0max,
        )

        print("Iteration ", c, " amu iv = ", amu_iv * 1e10)

        e_n, asq_n = solve_ll(
            8,
            L,
            m_pi,
            phase_shift=create_phase_instance,
            absfpi=create_absFpi_instance,
            phase_args=(m_pi, m_rho, par, "uniform", random_integers[c]),
            absfpi_args=(m_pi, m_rho, par, "uniform", random_integers[c]),
        )

        amu_fv, err_amu_fv = mll_amu(m_muon, e_n, asq_n, x0min=x0min, x0max=x0max)
        print("Iteration ", c, " amu fv = ", amu_fv * 1e10, "+- ", err_amu_fv * 1e10)

        amu_iv *= 1e10
        amu_fv *= 1e10
        err_amu_fv *= 1e10
        diff = amu_iv - amu_fv
        print("Iteration ", c, "finiteL diff = ", diff)

        amu_iv_list.append(amu_iv)
        amu_fv_list.append(amu_fv)
        amu_diff_list.append(diff)
        amu_err_list.append(err_amu_fv)

        result_sample = {
            "Sample": {
                "n": int(c),
                "seed": int(random_integers[c]),
                "delta_L_amu": diff,
                "amu_iv": amu_iv,
                "amu_fv": amu_fv,
                "truncation_error": err_amu_fv,
            }
        }
        with open(file_path, "a") as json_file:
            json.dump(result_sample, json_file, indent=4)

    result_error = {
        "ParErrors": {
            "delta_l_amu_AVG": np.mean(amu_diff_list),
            "delta_l_amu_ERR": np.std(amu_diff_list),
            "amu_iv_AVG": np.mean(amu_iv_list),
            "amu_iv_ERR": np.std(amu_iv_list),
            "amu_fv_AVG": np.mean(amu_fv_list),
            "amu_fv_ERR": np.std(amu_fv_list),
            "trunc_err_AVG": np.mean(amu_err_list),
            "trunc_err_ERR": np.std(amu_err_list),
        }
    }

    with open(file_path, "a") as json_file:
        json.dump(result_error, json_file, indent=4)

    return


if __name__ == "__main__":
    x0_min = 1.628 * gev_fm_conversion
    x0_max = np.inf
    L = Lref_fm * gev_fm_conversion
    m_pi = mass_pi0_GeV
    m_muon = mass_muon_GeV
    m_rho = mass_rho0_GeV
    par = par_CFD_2sigma
    n_copies = 3

    data = {
        "Inputs": {
            "Method": "MLL",
            "Parametrisation": "GKPRY",
            "L_[fm]": L / gev_fm_conversion,
            "m_pi_[GeV]": m_pi,
            "m_rho_[GeV]": m_rho,
            "x0_min_[fm]": x0_min / gev_fm_conversion,
            "x0_max_[fm]": x0_max,
            "phase_shift_parameters": par,
            "ncopies": n_copies,
        }
    }

    x0string = "{:2.3f}".format(x0_min)
    file_path = (
        "GKPRY_"
        + "MLL_"
        + "x0min_"
        + "{:2.3f}".format(x0_min / gev_fm_conversion)
        + "_Nsamples_"
        + str(n_copies)
        + "_L_"
        + str(L / gev_fm_conversion)
        + "_mpi_"
        + str(m_pi)
        + ".json"
    )

    print("Output file ", file_path)

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    delta_amu, amu_iv, amu_fv, trunc = mll_central(
        L, m_muon, m_pi, m_rho, par=par, x0min=x0_min, x0max=x0_max
    )

    result_central = {
        "Results": {
            "delta_L_amu": delta_amu,
            "amu_iv": amu_iv,
            "amu_fv": amu_fv,
            "truncation_error": trunc,
        }
    }

    with open(file_path, "a") as json_file:
        json.dump(result_central, json_file, indent=4)

    mll_py_err(
        file_path,
        L,
        n_copies,
        mass_muon_GeV,
        mass_pi0_GeV,
        mass_rho0_GeV,
        par,
        x0min=x0_min,
        x0max=x0_max,
    )

    exit()
