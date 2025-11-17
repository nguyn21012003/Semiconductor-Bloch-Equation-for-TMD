import csv

import numpy as np
from numpy import pi, sqrt
from tqdm import tqdm

from core.genHam import tbm_Hamiltonian
from settings.configs import Config

from core import genGrid


def writeBandstructure(data, modelNeighbor, file):
    karr, eigenValues = calcBandstructure(data, modelNeighbor)
    N = Config.N
    with open(file, "w", newline="") as f:
        header = ["kx", "L1", "L2", "L3"]
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for i in tqdm(range(N), desc="Write file"):
            row = {
                "kx": karr[i, i, 0],
                "L1": eigenValues[i, i, 0],
                "L2": eigenValues[i, i, 1],
                "L3": eigenValues[i, i, 2],
            }

            writer.writerow(row)
    return None


def calcBandstructure(data, modelNeighbor):
    N = Config.N
    alattice = data["alattice"]
    # grid K phai chay lai moi lan do khac alattice
    karr, dkx, dky = genGrid.Monkhorst(alattice, N)
    e_charge = Config.e_charge
    varepsilon = Config.varepsilon
    varepsilon0 = Config.varepsilon0

    V_const = e_charge**2 / (2 * varepsilon * varepsilon0) * dkx * dky / (2 * pi) ** 2
    print(V_const)

    eigenValues = np.zeros([N, N, 3])
    for i in tqdm(range(N), desc="Calculate bandstructure"):
        for j in range(N):
            alpha = karr[i, j, 0] / 2 * alattice
            beta = sqrt(3) / 2 * karr[i, j, 1] * alattice
            ham, dhkx, dhky = tbm_Hamiltonian(alpha, beta, data, modelNeighbor, alattice)
            vals = np.linalg.eigvalsh(ham)
            eigenValues[i, j, :] = vals
    return karr, eigenValues
