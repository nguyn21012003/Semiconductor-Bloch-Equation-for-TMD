import numpy as np
from numpy import pi, sqrt
from tqdm import tqdm

from core.genHam import tbm_Hamiltonian

from settings.configs import Config

hbar = Config.hbar
m_e = Config.m_e


def calcMass(data: dict, modelNeighbor: str):
    alattice = data["alattice"] * 1e-10
    G = 4 * pi / (sqrt(3) * alattice)
    dk = G / 500
    w = np.zeros([3, 3, 3])
    meffx = np.zeros(3)
    meffy = np.zeros(3)

    for i in tqdm(range(3)):
        for j in range(3):
            kx = 4 * pi / (3 * alattice) + (i - 1) * dk
            ky = (j - 1) * dk
            alpha = kx / 2 * alattice
            beta = sqrt(3) / 2 * ky * alattice

            hamiltonian = tbm_Hamiltonian(alpha, beta, data, modelNeighbor)
            vals = np.linalg.eigvalsh(hamiltonian)
            for k in range(3):
                w[k, i, j] = vals[k] * 1.602176634e-19
    for k in range(3):
        d2ex = (w[k, 2, 1] - 2 * w[k, 1, 1] + w[k, 0, 1]) / (dk**2)
        d2ey = (w[k, 1, 2] - 2 * w[k, 1, 1] + w[k, 1, 0]) / (dk**2)
        meffx[k] = (hbar**2) / (d2ex * m_e)
        meffy[k] = (hbar**2) / (d2ey * m_e)

    meff_e = round(meffx[1], 4)
    meff_h = round(abs(meffx[0]), 4)
    mr = round(meff_e * meff_h / (meff_e + meff_h), 4)

    return meff_e, meff_h, mr
