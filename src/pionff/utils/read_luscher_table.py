import numpy as np
import os


def read_luscher_function():
    """
    Reads tabulated values of the functions needed to
    solve Lusher quantisation condition.
    returns:
    q^2, phi(q), tan(phi), Zeta_{00}
    where q is the dimensionless momentum
    """
    column1 = []
    column2 = []
    column3 = []
    column4 = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "luscher_input.dat")
    with open(file_path, "r") as file:
        # Skip the header
        next(file)
        for line in file:
            # split the line by whitespace
            cols = line.split()
            col1, col2, col3, col4 = cols
            column1.append(float(col1))
            column2.append(float(col2))
            column3.append(float(col3))
            column4.append(float(col4))

    qsq = np.array(column1)  #   q^2
    phi_q_normalised = np.array(column2)  #   phi(q)/(pi*q^2)
    tan_phi = np.array(column3)  #   tan(phi(q))
    zeta00 = np.array(column4)  #   Z_{00}(1,q^2)

    return qsq, phi_q_normalised, tan_phi, zeta00
