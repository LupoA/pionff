import numpy as np
import json
from pionff.hp.fv_corrections import correction_amu
from pionff.formfactors.gkpry import phase_shift as gkpry_delta
from pionff.formfactors.gkpry import (
    par_CFD_2sigma,
    create_phase_instance,
)
from pionff.params import (
    mass_pi0_GeV,
    Lref_fm,
    gev_fm_conversion,
    mass_rho0_GeV,
    mass_muon_GeV,
)


def gkpry_central(file_path, L, m_muon, m_pi, m_rho, par, x0min, x0max):
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

    result_central = {
        "Results": {
            "delta_L_amu": delta_amu * 1e10,
            "truncation_error": err * 1e10,
        }
    }
    with open(file_path, "a") as json_file:
        json.dump(result_central, json_file, indent=4)

    return


def gkpry_errors(file_path, L, m_muon, m_pi, m_rho, par, n_copies, x0min, x0max):
    assert x0min == 0

    random_int = np.random.randint(1, 1e5, n_copies)
    res_list = []
    err_list = []

    for c in range(n_copies):
        delta_amu, err = correction_amu(
            x0max,
            L,
            m_muon,
            m_pi,
            create_phase_instance,
            m_pi,
            m_rho,
            par,
            "uniform",
            random_int[c],
        )
        print("Iteraction ", c, "res = ", delta_amu, " +- ", err)

        result_sample = {
            "Sample": {
                "n": int(c),
                "seed": int(random_int[c]),
                "delta_L_amu": delta_amu * 1e10,
                "truncation_error": err * 1e10,
            }
        }
        with open(file_path, "a") as json_file:
            json.dump(result_sample, json_file, indent=4)

        res_list.append(delta_amu)
        err_list.append(err)

    result_error = {
        "ParErrors": {
            "delta_l_amu_AVG": np.mean(res_list) * 1e10,
            "delta_l_amu_ERR": np.std(res_list) * 1e10,
            "trunc_err_AVG": np.mean(err_list) * 1e10,
            "trunc_err_ERR": np.std(err_list) * 1e10,
        }
    }

    with open(file_path, "a") as json_file:
        json.dump(result_error, json_file, indent=4)

    print("avg res = ", np.mean(res_list) * 1e10)
    print("avg trunc err = ", np.mean(err_list) * 1e10)
    print("par error = ", np.std(res_list) * 1e10)
    print("par error on trunc = ", np.std(err_list) * 1e10)
    print("\n")
    print(res_list)
    return


if __name__ == "__main__":
    x0_min = 0 * gev_fm_conversion
    x0_max = (1.682) * gev_fm_conversion
    L = Lref_fm * gev_fm_conversion
    m_pi = mass_pi0_GeV
    m_muon = mass_muon_GeV
    m_rho = mass_rho0_GeV
    par = par_CFD_2sigma
    n_copies = 10

    inputs = {
        "Inputs": {
            "Method": "HP",
            "Parametrisation": "GKPRY",
            "L_[fm]": L / gev_fm_conversion,
            "m_pi_[GeV]": m_pi,
            "m_rho_[GeV]": m_rho,
            "x0_min_[fm]": x0_min / gev_fm_conversion,
            "x0_max_[fm]": x0_max / gev_fm_conversion,
            "phase_shift_parameters": par,
            "ncopies": n_copies,
        }
    }

    file_path = (
        "GKPRY_"
        + "HP_"
        + "x0min_"
        + "{:2.3f}".format(x0_min / gev_fm_conversion)
        + "x0max_"
        + "{:2.3f}".format(x0_max / gev_fm_conversion)
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
        json.dump(inputs, json_file, indent=4)

    gkpry_central(file_path, L, m_muon, m_pi, m_rho, par, x0_min, x0_max)

    gkpry_errors(file_path, L, m_muon, m_pi, m_rho, par, n_copies, x0_min, x0_max)

    exit()
