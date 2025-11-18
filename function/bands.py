import csv

import numpy as np
from numpy import pi, sqrt
from tqdm import tqdm

from core.genHam import tbm_Hamiltonian
from settings.configs import Config

from core import genGrid


cfg = Config()
base_dir = cfg.base_dir


def writeBandstructure(data, name, modelNeighbor, file):
    N = Config.N
    karr, eigenValues, eigenValues_up, eigenValues_down = calcBandstructure(data, modelNeighbor, N)
    dir_path = f"{base_dir}/{name}"
    file_up = f"{dir_path}/bandstructure_u.dat"
    file_down = f"{dir_path}/bandstructure_d.dat"

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

    with open(file_up, "w", newline="") as fu:
        header = ["kx", "L1", "L2", "L3"]
        up = csv.DictWriter(fu, fieldnames=header)
        up.writeheader()
        for i in tqdm(range(N), desc=f"Writing {file_up}"):
            row = {
                "kx": karr[i, i, 0],
                "L1": eigenValues_up[i, i, 0],
                "L2": eigenValues_up[i, i, 1],
                "L3": eigenValues_up[i, i, 2],
            }
            up.writerow(row)

    with open(file_down, "w", newline="") as fd:
        header = ["kx", "L1", "L2", "L3"]
        down = csv.DictWriter(fd, fieldnames=header)
        down.writeheader()
        for i in tqdm(range(N), desc=f"Writing {file_down}"):
            row = {
                "kx": karr[i, i, 0],
                "L1": eigenValues_down[i, i, 0],
                "L2": eigenValues_down[i, i, 1],
                "L3": eigenValues_down[i, i, 2],
            }
            down.writerow(row)
    return None


def calcBandstructure(data, modelNeighbor, N):
    alattice = data["alattice"]
    # grid K phai chay lai moi lan do khac alattice
    karr, dkx, dky = genGrid.Monkhorst(alattice, N)

    eigenValues = np.zeros((N, N, 3))
    eigenValues_up = np.zeros((N, N, 3))
    eigenValues_down = np.zeros((N, N, 3))
    for i in tqdm(range(N), desc="Calculate bandstructure"):
        for j in range(N):
            alpha = karr[i, j, 0] / 2 * alattice
            beta = sqrt(3) / 2 * karr[i, j, 1] * alattice
            ham, dhkx, dhky, hamu, hamd = tbm_Hamiltonian(alpha, beta, data, modelNeighbor, alattice)
            vals = np.linalg.eigvalsh(ham)
            eigenValues[i, j, :] = vals

            vals_up = np.linalg.eigvalsh(hamu)
            eigenValues_up[i, j, :] = vals_up

            vals_down = np.linalg.eigvalsh(hamd)
            eigenValues_down[i, j, :] = vals_down

    return karr, eigenValues, eigenValues_up, eigenValues_down
