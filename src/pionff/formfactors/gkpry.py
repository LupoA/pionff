import numpy as np
import random
from pionff.utils.kinematics import e_to_k_2particles
from pionff.formfactors.omnes import omnes_function
from pionff.formfactors.infinite_volume import (
    a_mu_from_rho_iv,
    corr_iv,
    spectral_density_iv,
)
from pionff.utils.debug_opt import timeit
from pionff.params import DEBUG_MODE


#   Ref: [https://arxiv.org/pdf/1102.2183.pdf]

par_UFD = {
    "b0": 1.055,
    "b1": 0.15,
    "b0_err": 0.011,
    "b1_err": 0.05,
    "lambda_1": 1.57,
    "lambda_1_err": 0.18,
    "lambda_2": -1.96,
    "lambda_2_err": 0.49,
}

par_CFD = {
    "b0": 1.043,
    "b1": 0.19,
    "b0_err": 0.011,
    "b1_err": 0.05,
    "lambda_1": 1.39,
    "lambda_1_err": 0.18,
    "lambda_2": -1.70,
    "lambda_2_err": 0.49,
}

m_kaon = 0.497611  #   GeV
e_intermediate = 2 * m_kaon  #   GeV
e_asymptotic = 1.420  #   GeV


def _w(e, e0):
    return (e - np.sqrt((e0 * e0) - (e * e))) / (e + np.sqrt((e0 * e0) - (e * e)))


def _cot_delta_low(e, m_pi, m_rho, par):
    """
    Inpus must be in GeV
    """
    assert np.all(
        e <= e_intermediate
    ), "This expression for the phase shift is only valid up to e_intermediate = 2m_kaon"
    b0 = par["b0"]
    b1 = par["b1"]
    e0 = 1.05  #   GeV
    k = e_to_k_2particles(e, m_pi)
    res = ((m_rho * m_rho) - (e * e)) * e / (2 * k * k * k)
    res *= b0 + (_w(e, e0) * b1) + (2 * m_pi * m_pi * m_pi / (m_rho * m_rho * e))
    return res


def _tan_delta_low(e, m_pi, m_rho, par):
    """
    Inpus must be in GeV
    """
    assert np.all(
        e <= e_intermediate
    ), "This expression for the phase shift is only valid up to e_intermediate = 2m_kaon"
    b0 = par["b0"]
    b1 = par["b1"]
    e0 = 1.05  # GeV
    k = e_to_k_2particles(e, m_pi)
    res = (2 * k * k * k) / (((m_rho * m_rho) - (e * e)) * e)
    res /= b0 + (_w(e, e0) * b1) + (2 * m_pi * m_pi * m_pi / (m_rho * m_rho * e))
    return res
    # assert np.all(e <= e_intermediate), "This expression for the phase shift is only valid up to e_intermediate = 2m_kaon"
    # res = 1 / _cot_delta_low(e, m_pi, m_rho, par)
    # return res


def _delta_low(e, m_pi, m_rho, par):
    return np.arctan(_tan_delta_low(e, m_pi, m_rho, par))


def _delta_intermediate(e, m_pi, m_rho, par):
    assert np.all(
        (e > e_intermediate) & (e <= e_asymptotic)
    ), "Expression valid between e_intermediate = 2m_kaon and 1420 MeV"
    lambda_1 = par["lambda_1"]
    lambda_2 = par["lambda_2"]
    lambda_0 = _delta_low(e_intermediate, m_pi, m_rho, par)
    _term = (e / (2 * m_kaon)) - 1
    return lambda_0 + (lambda_1 * _term) + lambda_2 * (_term**2)


def _delta_asymptotic(e, m_pi, m_rho, par):
    """
    not part of the reference, continuously extends the phase shift up to 180 degrees
    """
    assert np.all(e > e_asymptotic), "Expression valid above 1.420 GeV"
    d_sa = _delta_intermediate(e_asymptotic, m_pi, m_rho, par)
    if d_sa < 0:
        d_sa += np.pi
    return np.pi + (
        (d_sa - np.pi) * 2 / (1 + ((e * e) / (e_asymptotic * e_asymptotic)) ** (3 / 4))
    )


def phase_shift(e, m_pi, m_rho, par=par_CFD):
    if isinstance(e, (float, int)):
        if e <= e_intermediate and e >= 2 * m_pi:
            result = _delta_low(e, m_pi, m_rho, par)
        elif e > e_intermediate and e <= e_asymptotic:
            result = _delta_intermediate(e, m_pi, m_rho, par)
        elif e > e_asymptotic:
            result = _delta_asymptotic(e, m_pi, m_rho, par)
        elif e < 2 * m_pi:
            result = 0
        else:
            raise ValueError("Phase shift is not defined at energy = ", e)

        if result < 0:
            result += np.pi
        return result
    else:
        conditions = [
            (e <= e_intermediate) & (e >= 2 * m_pi),
            (e > e_intermediate) & (e <= e_asymptotic),
            e > e_asymptotic,
            e < 2 * m_pi,
        ]

        result = np.empty_like(e)
        result[conditions[0]] = _delta_low(e[conditions[0]], m_pi, m_rho, par)
        result[conditions[1]] = _delta_intermediate(e[conditions[1]], m_pi, m_rho, par)
        result[conditions[2]] = _delta_asymptotic(e[conditions[2]], m_pi, m_rho, par)
        result[conditions[3]] = 0
        result[result < 0] += np.pi

        return result


@timeit(DEBUG_MODE)
def phase_shift_errors(e_values, m_pi, m_rho, par, n_copies=100, error="max"):
    """
    In order to vary all parameters simulatenously within errors,
    values are generated according to uniform distributions
    """
    results_std = []

    # we allow the function to work on e_values being a float or a np.array
    # by making it always a np.array
    if not isinstance(e_values, np.ndarray):
        e_values = np.array([e_values])

    for e in e_values:
        results = []

        for _ in range(n_copies):
            updated_par = par.copy()

            # vary b0
            b0 = par["b0"]
            b0_err = par["b0_err"]
            updated_par["b0"] = random.uniform(b0 - b0_err, b0 + b0_err)

            # vary b1
            b1 = par["b1"]
            b1_err = par["b1_err"]
            updated_par["b1"] = random.uniform(b1 - b1_err, b1 + b1_err)

            # vary lambda_1
            lambda_1 = par["lambda_1"]
            lambda_1_err = par["lambda_1_err"]
            updated_par["lambda_1"] = random.uniform(
                lambda_1 - lambda_1_err, lambda_1 + lambda_1_err
            )

            # vary lambda_2
            lambda_2 = par["lambda_2"]
            lambda_2_err = par["lambda_2_err"]
            updated_par["lambda_2"] = random.uniform(
                lambda_2 - lambda_2_err, lambda_2 + lambda_2_err
            )

            # run
            result = phase_shift(e, m_pi, m_rho, updated_par)
            results.append(result)

        if error == "max":
            _std = abs(np.amax(results) - np.amin(results))
            results_std.append(_std)
        elif error == "std":
            results_std.append(np.std(results))
        else:
            raise ValueError("Allowed error types are 'max' and 'std'.")

    return np.array(results_std)


########################################


def argFpi(e, m_pi, m_rho):
    return phase_shift(e, m_pi, m_rho)


def absFpi_of_s(s, m_pi, m_rho):
    return omnes_function(s, 4 * m_pi * m_pi, phase_shift, m_pi, m_rho)


def absFpi(e, m_pi, m_rho):
    return absFpi_of_s(e * e, m_pi, m_rho)


def py_spectral_density(e, m_pi, m_rho):
    """
    HVP spectral density (infinite volume); dimensionless
    """
    return spectral_density_iv(e, m_pi, absFpi, m_pi, m_rho)


def py_corr(t, m_pi, m_rho):
    """
    \int_0^inf rho(E) E^2 exp(-tE) ::: has dimension [E]^3
    """
    return corr_iv(t, m_pi, absFpi, m_pi, m_rho)


def py_amu(mass_muon, m_pi, m_rho, x0min=0, x0max=np.inf):
    result = a_mu_from_rho_iv(
        mass_muon, m_pi, absFpi, m_pi, m_rho, x0min=x0min, x0max=x0max
    )
    return result
