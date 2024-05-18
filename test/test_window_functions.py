from pionff.utils.windows import StandardWindows
import numpy as np
from scipy.integrate import quad


def test_integrated_windows():
    window_t = StandardWindows()

    def _integrand(t):
        return np.sqrt(t)

    reference = quad(_integrand, a=0, b=4)
    print(reference[0], reference[1])

    def _integrand_SD(t):
        return _integrand(t) * window_t.sd(t, units_fm=True)

    def _integrand_ID(t):
        return _integrand(t) * window_t.id(t, units_fm=True)

    def _integrand_LD(t):
        return _integrand(t) * window_t.ld(t, units_fm=True)

    value_sd = quad(_integrand_SD, a=0, b=4)
    value_id = quad(_integrand_ID, a=0, b=4)
    value_ld = quad(_integrand_LD, a=0, b=4)

    print(
        value_ld[0] + value_sd[0] + value_id[0], value_ld[1] + value_sd[1] + value_id[1]
    )

    diff = abs(value_ld[0] + value_sd[0] + value_id[0] - reference[0])
    assert diff < abs(value_ld[1] + value_sd[1] + value_id[1])


test_integrated_windows()
