import numpy as np


def k_to_E_2particles(k, m):
    return 2 * np.sqrt((m * m) + (k * k))


def e_to_k_2particles(e, m):
    return np.sqrt((e * e / 4) - (m * m))
