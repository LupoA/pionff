import numpy as np
from pionff.utils.kinematics import e_to_k_2particles


"""https://arxiv.org/pdf/1908.00921"""
par_DHMZ_2019 = {
    "b0": 1.03954085,
    "b0err": 0.000309867848 * 10,
    "b1": -0.132091806,
    "b1err": 0.0110540115 * 10,
    "mrho": 0.77449101,
    "mrho_err": 8.43231604e-05 * 10,
    "corr_b0_b1": 0.394912587,
    "corr_b0_mrho": -0.402562365,
    "corr_b1_mrho": -0.962100563,
}


class DHMZpar:
    def __init__(self, par):
        """
        auxiliary class to automatically fill the covariance
        """
        self.b0 = par["b0"]
        self.b1 = par["b1"]
        self.mrho = par["mrho"]
        self.b0_sigma = par["b0err"]
        self.b1_sigma = par["b1err"]
        self.mrho_sigma = par["mrho_err"]
        self.cov = np.zeros((3, 3))
        self.cov[0][0] = par["b0err"] ** 2
        self.cov[1][1] = par["b1err"] ** 2
        self.cov[2][2] = par["mrho_err"] ** 2
        self.cov[0][1] = par["corr_b0_b1"] * par["b0err"] * par["b1err"]
        self.cov[0][2] = par["corr_b0_mrho"] * par["b0err"] * par["mrho_err"]
        self.cov[1][2] = par["corr_b1_mrho"] * par["b1err"] * par["mrho_err"]
        self.cov[1][0] = self.cov[0][1]
        self.cov[2][0] = self.cov[0][2]
        self.cov[2][1] = self.cov[1][2]


e0 = 1.3  # GeV
e_matching = 1.3  # GeV


def _w(e, e0):
    return (e - np.sqrt((e0 * e0) - (e * e))) / (e + np.sqrt((e0 * e0) - (e * e)))


def _cot_delta_low(e, m_pi, par):
    """
    Valid up to 1.3 GeV
    """
    assert np.all(
        e <= 1.3
    ), "This expression for the phase shift is only valid up to 1.3 GeV "
    b0 = par["b0"]
    b1 = par["b1"]
    m_rho = par["mrho"]
    k = e_to_k_2particles(e, m_pi)
    res = ((m_rho * m_rho) - (e * e)) * e / (2 * k * k * k)
    res *= b0 + (_w(e, e0) * b1) + (2 * m_pi * m_pi * m_pi / (m_rho * m_rho * e))
    return res


def _tan_delta_low(e, m_pi, par):
    """
    Inpus must be in GeV
    """
    assert np.all(
        e <= 1.3
    ), "This expression for the phase shift is only valid up to 1.3 GeV "
    b0 = par["b0"]
    b1 = par["b1"]
    m_rho = par["mrho"]
    k = e_to_k_2particles(e, m_pi)
    res = (2 * k * k * k) / (((m_rho * m_rho) - (e * e)) * e)
    res /= b0 + (_w(e, e0) * b1) + (2 * m_pi * m_pi * m_pi / (m_rho * m_rho * e))
    return res


def _delta_low(e, m_pi, par):
    return np.arctan(_tan_delta_low(e, m_pi, par))


def _delta_asymptotic(e, m_pi, par):
    """
    not part of the reference, continuously extends the phase shift up to 180 degrees
    """
    assert np.all(e > 1.3), "Expression valid above 1.3 GeV"
    d_sa = _delta_low(e_matching, m_pi, par)
    if d_sa < 0:
        d_sa += np.pi
    return np.pi + (
        (d_sa - np.pi) * 2 / (1 + ((e * e) / (e_matching * e_matching)) ** (3 / 4))
    )


def phase_shift(e, m_pi, par):
    if isinstance(e, (float, int)):
        if e <= e_matching and e >= 2 * m_pi:
            result = _delta_low(e, m_pi, par)
        elif e > e_matching:
            result = _delta_asymptotic(e, m_pi, par)
        elif e < 2 * m_pi:
            result = 0
        else:
            raise ValueError("Phase shift is not defined at energy = ", e)

        if result < 0:
            result += np.pi
        return result
    else:
        conditions = [
            (e <= e_matching) & (e >= 2 * m_pi),
            e > e_matching,
            e < 2 * m_pi,
        ]

        result = np.zeros_like(e, dtype=float)
        result[conditions[0]] = _delta_low(e[conditions[0]], m_pi, par)
        result[conditions[1]] = _delta_asymptotic(e[conditions[1]], m_pi, par)
        result[conditions[2]] = 0
        result[result < 0] += np.pi

        return result


def create_par_toys(par, size=100, seed=42, return_cov=False):
    """
    creates multivariate gaussian distribution for [b0, b1, m_rho]
    given the covariance and mean values present in par
    """
    par_t = DHMZpar(par)
    mean = [par_t.b0, par_t.b1, par_t.mrho]
    covariance = par_t.cov

    np.random.seed(seed)
    samples = np.random.multivariate_normal(mean, covariance, size=size)

    if not return_cov:
        return samples
    else:
        return samples, covariance


def phase_errors(e_range, samples, m_pi):
    phase_instances = []

    for n in range(len(samples[:, 0])):
        _par = {
            "b0": samples[n, 0],
            "b1": samples[n, 1],
            "mrho": samples[n, 2],
        }

        phase_instances.append(phase_shift(e_range, m_pi, _par))

    mean_phase = np.mean(phase_instances, axis=0)
    std_phase = np.std(phase_instances, axis=0)

    return mean_phase, std_phase
