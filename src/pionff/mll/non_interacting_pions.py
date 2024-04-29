import numpy as np


def free_fv_energies_and_amplitudes(m_pi, L, qmax=8):
    """
    energies and matrix elements of two non-interacting pions in a finite box
    """

    def compress_matrix_elements(vector):
        compressed_vector = []
        current_sum = vector[0]

        for i in range(1, len(vector)):
            if vector[i] == vector[i - 1]:
                current_sum += vector[i]
            else:
                compressed_vector.append(current_sum)
                current_sum = vector[i]

        compressed_vector.append(current_sum)

        return compressed_vector

    def compress_energies(vector):
        compressed_vector = []
        for i in range(len(vector)):
            if i == 0 or vector[i] != vector[i - 1]:
                compressed_vector.append(vector[i])
        return compressed_vector

    e_n = []
    asq_n = []

    for q1 in range(-qmax, qmax):
        for q2 in range(-qmax, qmax):
            for q3 in range(-qmax, qmax):
                qsq = q1**2 + q2**2 + q3**2
                ksq = qsq * (2 * np.pi / L) * (2 * np.pi / L)
                energy = 2 * np.sqrt(m_pi**2 + ksq)
                e_n.append(energy)
                ansq = ksq / (energy * energy)
                ansq /= L * L * L
                ansq *= 4
                asq_n.append(ansq)

    sorted_indices = sorted(range(len(e_n)), key=lambda k: e_n[k])
    sorted_e_n = [e_n[i] for i in sorted_indices]
    sorted_a_n = [asq_n[i] for i in sorted_indices]

    sorted_a_n = np.array(sorted_a_n)
    sorted_e_n = np.array(sorted_e_n)

    sorted_e_n = sorted_e_n[1:]  # Remove qsq = 0
    sorted_a_n = sorted_a_n[1:]  # Remove qsq = 0

    reduced_a_sq_n = compress_matrix_elements(sorted_a_n)
    reduced_e_n = compress_energies(sorted_e_n)

    return reduced_e_n, reduced_a_sq_n
