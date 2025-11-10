from dataclasses import dataclass
from datetime import datetime

from numpy import pi
import numpy as np


@dataclass
class Config:
    time_run = datetime.now().strftime("%a-%m-%d")
    neighbor = "NN"
    approx = "GGA"
    base_dir = f"./results/{time_run}/{neighbor}"
    h = 6.62607007e-34  # kg m**2 / s**2
    hbar = h / (2 * pi)
    m_e = 9.10938356e-31  # kg
    e_charge = 1.602176621e-19  # Coulomb
    phi_0 = h / e_charge
    isMatrixTransform = True
    qmax = 797  # default qmax
    varepsilon = 2.5  # const
    varepsilon0 = 8.8541878128e-12  # F/m
    N = 101
    rkcutoff = 2.0
    detuning = 1e-3  # eV

    ###################### Parameter for pulse
    E0 = 1e6 * 1e-7  # V/nm
    time_duration = 5  # fs
    epl = 0  # Linear polarized
    tmin = -4 * (time_duration / np.sqrt(2 * np.log(2)))  # fs
    tmax = 1000.0  # fs
    dt = 1.0 / 50.0  # fs
    ntmax = int((tmax - tmin) / dt)
