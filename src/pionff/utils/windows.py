import numpy as np
from pionff.params import gev_fm_conversion


def ctheta(t, tp, delta):
    return (np.tanh((t - tp) / delta) + 1) / 2


def sd_window(t, t0, delta):
    return 1 - ctheta(t, t0, delta)


def id_window(t, t0, t1, delta):
    return ctheta(t, t0, delta) - ctheta(t, t1, delta)


def ld_window(t, t1, delta):
    return ctheta(t, t1, delta)


class StandardWindows:
    def __init__(self):
        self.t0_fm = 0.4
        self.t1_fm = 1.0
        self.t_far_fm = 3.5
        self.Delta_fm = 0.15
        self.t0_gev = 0.4 * gev_fm_conversion
        self.t1_gev = 1.0 * gev_fm_conversion
        self.t_far_gev = 3.5 * gev_fm_conversion
        self.Delta_gev = 0.15 * gev_fm_conversion

        self.t_dd_fm = 2.8
        self.t_dd_gev = 2.8 * gev_fm_conversion

    def sd(self, t, units_fm: bool):
        if not units_fm:
            return sd_window(t, self.t0_gev, self.Delta_gev)
        else:
            return sd_window(t, self.t0_fm, self.Delta_fm)

    def id(self, t, units_fm: bool):
        if not units_fm:
            return id_window(t, self.t0_gev, self.t1_gev, self.Delta_gev)
        else:
            return id_window(t, self.t0_fm, self.t1_fm, self.Delta_fm)

    def ld(self, t, units_fm: bool):
        if not units_fm:
            return ld_window(t, self.t1_gev, self.Delta_gev)
        else:
            return ld_window(t, self.t1_fm, self.Delta_fm)

    def ol(self, t, units_fm: bool):
        if not units_fm:
            return id_window(t, self.t_dd_gev, self.t_far_gev, self.Delta_gev)
        else:
            return id_window(t, self.t_dd_fm, self.t_far_fm, self.Delta_fm)
