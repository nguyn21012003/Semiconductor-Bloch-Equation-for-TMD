from dataclasses import dataclass
from datetime import datetime

from numpy import pi


@dataclass
class Config:
    time_run = datetime.now().strftime("%a-%m-%d")
    neighbor = "TNN"
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
    N = 201
    rkcutoff = 3.0 + 1e-7
