from dataclasses import dataclass
from datetime import datetime

from numpy import pi
import numpy as np


# @dataclass
# class Config:
#     time_run = datetime.now().strftime("%a-%m-%d")
#     neighbor = "NN"
#     approx = "GGA"
#     base_dir = f"./results/{time_run}/{neighbor}"
#     h = 6.62607007e-34  # kg m**2 / s**2
#     hbar = h / (2 * pi)
#     m_e = 9.10938356e-31  # kg
#     e_charge = 1.602176621e-19  # Coulomb
#     phi_0 = h / e_charge
#     isMatrixTransform = True
#     qmax = 797  # default qmax
#     varepsilon = 2.5  # const
#     varepsilon0 = 8.8541878128e-12  # F/m
#     N = 60
#     rkcutoff = 2.0
#     detuning = 1e-3  # eV
#     gamma = 65  # nm^3 / fs ==> # 6.5 * 1e-20 cm^3 / fs
#     T_cohenrent = 15  # fs day la (T2) trong bai cua thay Tuyen
#
#     ###################### Parameter for pulse
#     E0 = 1e6 * 1e-7  # V/nm
#     time_duration = 5  # fs
#     epl = 1  # Linear polarized
#     tmin = -4 * (time_duration / np.sqrt(2 * np.log(2)))  # fs
#     tmax = 1000.0  # fs
#     dt = 0.02  # fs


@dataclass
class Config:
    time_run = datetime.now().strftime("%a-%m-%d")
    neighbor = "TNN"
    approx = "GGA"
    base_dir = f"./results/{time_run}/{neighbor}"
    #################################################################
    h: float = 4.135667696  # Đơn vị: eV * fs (Planck's constant)
    hbar: float = h / (2 * pi)  # Đơn vị: eV * fs (Giá trị ~0.6582)
    m_e: float = 5.68563  # Đơn vị: eV*fs^2/nm^2
    e_charge: float = -1.0  # Đơn vị: e (điện tích cơ bản)
    phi_0: float = h / e_charge  # Đơn vị: eV*fs/e
    isMatrixTransform: bool = True
    qmax: int = 797  # default qmax
    varepsilon: float = 2.5  # Hằng số điện môi (không đơn vị)
    varepsilon0: float = 0.05526349  # Đơn vị: e^2/(eV*nm)
    N: int = 100  # Giống nkamax
    rkcutoff: float = 3.0
    detuning: float = 10e-3  # Đơn vị: eV
    gamma: float = 65.0  # Đơn vị: nm^3/fs
    T_cohenrent: float = 30.0  # Đơn vị: fs (T2)
    E0: float = 2.2e-3  # Đơn vị: V/nm (0.01)
    time_duration: float = 6.0  # Đơn vị: fs, hay còn gọi là bề rộng sung
    epl: float = 0.0  # Phân cực tròn (Circular polarized)
    tmin: float = -50  # Đơn vị: fs
    tmax: float = 1000.0  # Đơn vị: fs
    dt: float = 0.05  # Đơn vị: fs
