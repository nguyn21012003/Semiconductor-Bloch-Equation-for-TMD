import csv
from typing import Tuple

import numpy as np
from numpy import copy, pi, sqrt
from numpy.typing import NDArray
from tqdm import tqdm

from core import genGrid
from core.genHam import tbm_Hamiltonian
from settings.configs import Config


def Coulomb(data, modelNeighbor):
    alattice = data["alattice"] * 1e-1
    N = Config.N

    grid, dkx, dky = genGrid.Monkhorst(alattice, N)  # kArray is an array contain kx and ky with N,N,2 dimensions

    num_kcutoff, check_valid_cutoff_array = cutoff_Grid(N, grid, alattice)
    try:
        indices_k1_tuple = np.where(check_valid_cutoff_array[:, :, 0])  # check True elements and return index
        map_k1 = np.column_stack(indices_k1_tuple)
        indices_k2_tuple = np.where(check_valid_cutoff_array[:, :, 1])
        map_k2 = np.column_stack(indices_k2_tuple)
        mappingArray = np.vstack((map_k1, map_k2))
        if len(mappingArray) != num_kcutoff:
            raise ValueError(f"Error Map: ({len(mappingArray)}) and ({num_kcutoff})")

        print(f"Mapping done: {mappingArray.shape}")
    except ValueError as e:
        print(e)
        import sys

        sys.exit(1)

    eigenVectors = solveHamiltonian(data, grid, modelNeighbor)

    coulombInteractionArray = np.zeros([num_kcutoff, num_kcutoff, 2, 2], dtype=complex)
    vecs_at_cutoff = eigenVectors[mappingArray[:, 0], mappingArray[:, 1], :2, :]

    for i in tqdm(range(num_kcutoff), desc="Calculating Coulomb.."):
        vec_i = vecs_at_cutoff[i]  # (4, num_orbitals)
        vec_i_T = vec_i.T  # (num_orbitals, 4)
        is_k_i_in_K1 = i < len(map_k1)
        for j in range(num_kcutoff):
            is_k_j_in_K1 = j < len(map_k1)

            if is_k_i_in_K1 == is_k_j_in_K1:

                vec_j = vecs_at_cutoff[j]  # (4, num_orbitals)
                overlap = vec_j.conj() @ vec_i_T
                coulombInteractionArray[j, i, :, :] = overlap


def cutoff_Grid(N, grid, alattice) -> Tuple[int, NDArray[np.bool_]]:
    rkcutoff = Config.rkcutoff
    check_valid_cutoff_grid = np.zeros([N, N, 2], dtype=bool)
    count = 0
    for i in tqdm(range(N), desc="Cutting off K-points"):
        for j in range(N):
            rk1 = sqrt((grid[i, j, 0] - 4 * pi / (3 * alattice)) ** 2 + grid[i, j, 1] ** 2)
            rk2 = sqrt((grid[i, j, 0] + 4 * pi / (3 * alattice)) ** 2 + grid[i, j, 1] ** 2)
            if rk1 < rkcutoff:
                check_valid_cutoff_grid[i, j, 0] = True
                count += 1
            elif rk2 < rkcutoff:
                check_valid_cutoff_grid[i, j, 1] = True
                count += 1
    return count, check_valid_cutoff_grid


def solveHamiltonian(data, kArray, modelNeighbor):
    N = Config.N
    alattice = data["alattice"]
    # grid K phai chay lai moi lan do khac alattice

    eigenVectors = np.zeros([N, N, 3, 3], dtype=complex)
    for i in tqdm(range(N), desc="Calculate bandstructure"):
        for j in range(N):
            alpha = kArray[i, j, 0] / 2 * alattice
            beta = sqrt(3) / 2 * kArray[i, j, 1] * alattice
            ham = tbm_Hamiltonian(alpha, beta, data, modelNeighbor)
            _, vecs = np.linalg.eigh(ham)
            eigenVectors[i, j, :, :] = vecs[:, :3].T  # C[i, j, λ, orb]
            del _

    return eigenVectors
