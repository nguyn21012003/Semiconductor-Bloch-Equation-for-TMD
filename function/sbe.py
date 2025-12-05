import csv
from typing import Tuple

import numba as nb
import numpy as np
import pandas as pd
from numba import jit
from numpy import pi, sqrt
from numpy.linalg import norm
from numpy.typing import NDArray
from tqdm import tqdm

from core import genGrid, pulse
from core.genHam import tbm_Hamiltonian
from settings.configs import Config


def Hartree_Fock_func(data, modelNeighbor):
    alattice = data["alattice"]
    N = Config.N
    detuning = Config.detuning

    # grid, dkx, dky = genGrid.Rhombus(alattice, N)
    grid, dk2 = genGrid.Monkhorst(alattice, N)
    # grid, dkx, dky = genGrid.Cartesian(alattice, N)
    print(dk2)
    print(f"Grid dim:{grid.shape}")

    plot_grid(grid, "kgrid.dat")
    eigenValues, eigenVectors, pmx, pmy = solveHamiltonian(data, grid, modelNeighbor)

    egap = np.min(eigenValues[:, :, 2]) - np.max(eigenValues[:, :, 1])
    print(f"Energy gap: {egap} eV")
    print(f"Detuning: {detuning} eV")

    w0 = (egap) / Config.hbar
    print(f"Frequency: {w0}")

    check_degenaracy = np.zeros([6, 6])
    for i in tqdm(range(N), desc="Check degenaracy"):
        for j in range(N):
            for nu in range(0, 6):
                for mu in range(0, 6):
                    if np.abs(eigenValues[i, j, nu] - eigenValues[i, j, mu]) < 1e-2:
                        check_degenaracy[mu, nu] += 1

    # dk2 = dkx * dky * sqrt(3) / 2
    ################################################################# Post SBE
    p = (pmx, pmy)
    xi = dipole(eigenValues, p, check_degenaracy)

    ################################################################# SBE
    rho = np.zeros([N, N, 6, 6], dtype="complex")
    # note: chua co spin nen chi xet la 2, co spin thi la 4
    rho[:, :, 0, 0] = 1.0 + 0.0j
    rho[:, :, 1, 1] = 1.0 + 0.0j

    tmin = Config.tmin
    print("tmin:", tmin)
    tmax = Config.tmax
    print("tmax:", tmax)
    dt = Config.dt
    tL = Config.time_duration
    E0 = Config.E0
    ntmax = int((tmax - tmin) / dt)

    # data_Coulomb = Coulomb(alattice, eigenVectors, grid, dk2)
    CM = CoulMat(alattice, eigenVectors, grid, dk2)
    # data_Coulomb = None
    ############################################################ Calculating SBE
    P_t_array = np.zeros([ntmax, 2], dtype="complex")  # Save P(t)
    time_array = np.zeros(ntmax, dtype=float)
    Atx, Aty = 0, 0
    t = tmin
    for nt in tqdm(range(ntmax), desc="RK4 SBE: ", colour="blue"):
        Etx, Ety = pulse.Et(t, E0, tL)
        Etime = (Etx, Ety)
        Atx = Config.e_charge / Config.m_e * (Atx - Etx * dt)
        Aty = Config.e_charge / Config.m_e * (Aty - Ety * dt)
        Atime = (Atx, Aty)

        # rho = rk4(rho, t, dt, Etime, Atime, xi, p, dk2, data_Coulomb, eigenValues, w0)
        rho = rk4(rho, t, dt, Etime, Atime, xi, p, dk2, CM, eigenValues, w0)
        Pxt, Pyt = totalPolarizationFunction(rho, xi, dk2)

        P_t_array[nt, 0] = Pxt
        P_t_array[nt, 1] = Pyt
        time_array[nt] = t
        t += dt

    print(time_array)
    absorptionSpectra(P_t_array, egap, time_array)


def absorptionSpectra(P_t_array, egap, evoltime):
    hbar = Config.hbar
    dt = Config.dt
    E0 = Config.E0
    tL = Config.time_duration
    omegaL = egap / hbar
    omega = (egap - 1) / hbar
    domega = 0.002 / hbar
    output_file = "./results/AbSpect.csv"
    header = [
        "Energy_minus_Egap(eV)",
        "Re(Chi_x)",
        "Im(Chi_x)",
        "Re(Chi_y)",
        "Im(Chi_y)",
        "Re(E_w)",
        "Im(E_w)",
        "Alpha(w)",
    ]

    omega_array = np.arange(omega, (egap + 1) / hbar, domega)
    Ex_array = np.array([pulse.Et(t, E0, tL)[0] for t in evoltime])
    P_t_x = P_t_array[:, 0]
    P_t_y = P_t_array[:, 1]
    delta_omega_col = (omega_array - omegaL)[:, np.newaxis]
    eiwt_matrix = np.exp(1j * delta_omega_col * evoltime)

    s1_array = (eiwt_matrix @ P_t_x) * dt
    s2_array = (eiwt_matrix @ P_t_y) * dt
    s3_array = (eiwt_matrix @ Ex_array) * dt

    chi_x_array = s1_array / (s3_array + 1e-16)
    chi_y_array = s2_array / (s3_array + 1e-16)
    energy_col = omega_array * hbar - egap
    alpha_array = np.imag(chi_x_array + chi_y_array)
    # pmax_index = np.argmax(chi_x_array)
    # eBind = energy_col[pmax_index]
    # print("Exiton binding energy: ", eBind)

    data = {
        "Energy_minus_Egap(eV)": energy_col,
        "Re(Chi_x)": np.real(chi_x_array),
        "Im(Chi_x)": np.imag(chi_x_array),
        "Re(Chi_y)": np.real(chi_y_array),
        "Im(Chi_y)": np.imag(chi_y_array),
        "Re(E_w)": np.real(s3_array),
        "Im(E_w)": np.imag(s3_array),
        "Alpha(w)": alpha_array,
    }

    df = pd.DataFrame(data, columns=header)
    df.to_csv(output_file, index=False, float_format="%.10g")

    print(f"Đã ghi xong file: {output_file}")
    return True


def totalPolarizationFunction(rho, xi, dk2):
    sum1 = 0j
    sum2 = 0j

    ratio = dk2 / (4 * pi**2)
    for jb in [0, 1]:
        for ib in [2, 3, 4, 5]:
            sum1 += np.sum(xi[:, :, jb, ib, 0] * rho[:, :, ib, jb])
            sum2 += np.sum(xi[:, :, jb, ib, 1] * rho[:, :, ib, jb])

    Px = -sum1 * ratio
    Py = -sum2 * ratio
    return Px, Py


def rk4(rho, t, dt, Et, At, xi, p, dk22, exchange_term, E, w0):
    k1 = rhsSBE(rho, t, Et, At, xi, p, dk22, E, w0, exchange_term)
    k2 = rhsSBE(rho + 0.5 * k1 * dt, t, Et, At, xi, p, dk22, E, w0, exchange_term)
    k3 = rhsSBE(rho + 0.5 * k2 * dt, t, Et, At, xi, p, dk22, E, w0, exchange_term)
    k4 = rhsSBE(rho + 1.0 * k3 * dt, t, Et, At, xi, p, dk22, E, w0, exchange_term)

    return rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rhsSBE(rho, t, Et, At, xiov, p, dk2, E, omega_L, exchange_term):
    Etx, Ety = Et
    N = Config.N
    Y = np.zeros((N, N, 6, 6), dtype="complex")
    T2 = Config.T_cohenrent
    # hfse = hamiltonian_coulomb(exchange_term, rho, N)
    cm1, cm2, nkc = exchange_term
    hfse = HartreeFockSelfEnergy(rho, dk2, N, nkc, cm1, cm2)
    for v in [0, 1]:
        for c in [2, 3, 4, 5]:
            # ω_cv = (E_c − E_v)/ħ
            omegaCV = (E[:, :, c] - E[:, :, v]) / Config.hbar

            # dipole element ξ_cv
            xix = xiov[:, :, c, v, 0]
            xiy = xiov[:, :, c, v, 1]
            omegaRabi = (Etx * xix + Ety * xiy) / Config.hbar

            # HF term for (c,v)
            omegaHF = hfse[:, :, c, v] / Config.hbar

            # density matrix element ρ_cv
            rho_cv = rho[:, :, c, v]

            # equation of motion for ρ_cv
            rhodot = (
                -1j * (omegaCV - omega_L) * rho_cv
                + 1j * omegaHF
                - 1j * omegaRabi
                - rho_cv / T2
            )

            # store
            Y[:, :, c, v] = rhodot
            Y[:, :, v, c] = np.conjugate(rhodot)

    return Y


def plot_grid(kArr, file):
    kx = kArr[:, :, 0].flatten()
    ky = kArr[:, :, 1].flatten()
    np.savetxt(
        "./results/" + file, np.column_stack((kx, ky)), header="kx,ky", delimiter=","
    )


@jit(nopython=True, parallel=True)
def HartreeFockSelfEnergy(rho, dk2, n, nc, cm1, cm2):

    hfse = np.zeros((n, n, 6, 6), dtype=np.complex128)

    for j in nb.prange(nc + 1, 2 * nc + 1):
        for i in range(nc + 1, 2 * nc + 1):
            for l4 in [0, 1, 2, 3]:
                for l2 in [0, 1, 2, 3]:
                    for jj in range(nc + 1, 2 * nc + 1):
                        for ii in range(nc + 1, 2 * nc + 1):
                            for l1 in [0, 1]:
                                hfse[i - 1, j - 1, l2, l4] += -(
                                    cm1[
                                        (i - nc) + (j - nc - 1) * nc - 1,
                                        (ii - nc) + (jj - nc - 1) * nc - 1,
                                        l1,
                                        l2,
                                        l1,
                                        l4,
                                    ]
                                    * (1.0 - rho[ii - 1, jj - 1, l1, l1])
                                    * dk2
                                    / (4 * pi**2)
                                )
                            for l3 in [2, 3]:
                                hfse[
                                    i - 1,
                                    j - 1,
                                    l2,
                                    l4,
                                ] += (
                                    cm1[
                                        (i - nc) + (j - nc - 1) * nc - 1,
                                        (ii - nc) + (jj - nc - 1) * nc - 1,
                                        l3,
                                        l2,
                                        l3,
                                        l4,
                                    ]
                                    * rho[ii - 1, jj - 1, l3, l3]
                                    * dk2
                                    / (4 * pi**2)
                                )
                            for l1 in [0, 1, 2]:
                                for l3 in range(l1 + 1, 5):
                                    hfse[i - 1, j - 1, l2, l4] += cm1[
                                        (i - nc) + (j - nc - 1) * nc - 1,
                                        (ii - nc) + (jj - nc - 1) * nc - 1,
                                        l1,
                                        l2,
                                        l3,
                                        l4,
                                    ] * rho[ii - 1, jj - 1, l3, l1] * dk2 / (
                                        4 * pi**2
                                    ) + cm1[
                                        (i - nc) + (j - nc - 1) * nc - 1,
                                        (ii - nc) + (jj - nc - 1) * nc - 1,
                                        l3,
                                        l2,
                                        l1,
                                        l4,
                                    ] * rho[
                                        ii - 1,
                                        jj - 1,
                                        l1,
                                        l3,
                                    ] * dk2 / (
                                        4 * pi**2
                                    )

    for j in range(n - 2 * nc + 1, n - nc + 1):
        for i in range(n - 2 * nc + 1, n - nc + 1):
            for l4 in [0, 1, 2, 3]:
                for l2 in [0, 1, 2, 3]:
                    for jj in range(n - 2 * nc + 1, n - nc + 1):
                        for ii in range(n - 2 * nc + 1, n - nc + 1):
                            for l1 in [0, 1]:
                                hfse[i - 1, j - 1, l2, l4] += -(
                                    cm2[
                                        (i - n + 2 * nc)
                                        + (j - n + 2 * nc - 1) * nc
                                        - 1,
                                        (ii - n + 2 * nc)
                                        + (jj - n + 2 * nc - 1) * nc
                                        - 1,
                                        l1,
                                        l2,
                                        l1,
                                        l4,
                                    ]
                                    * (1.0 - rho[ii - 1, jj - 1, l1, l1])
                                    * dk2
                                    / (4 * pi**2)
                                )
                            for l3 in [2, 3]:
                                hfse[
                                    i - 1,
                                    j - 1,
                                    l2,
                                    l4,
                                ] += (
                                    cm2[
                                        (i - n + 2 * nc)
                                        + (j - n + 2 * nc - 1) * nc
                                        - 1,
                                        (ii - n + 2 * nc)
                                        + (jj - n + 2 * nc - 1) * nc
                                        - 1,
                                        l3,
                                        l2,
                                        l3,
                                        l4,
                                    ]
                                    * rho[ii - 1, jj - 1, l3, l3]
                                    * dk2
                                    / (4 * pi**2)
                                )
                            for l1 in [0, 1, 2]:
                                for l3 in range(l1 + 1, 5):
                                    hfse[i - 1, j - 1, l2, l4] += cm2[
                                        (i - n + 2 * nc)
                                        + (j - n + 2 * nc - 1) * nc
                                        - 1,
                                        (ii - n + 2 * nc)
                                        + (jj - n + 2 * nc - 1) * nc
                                        - 1,
                                        l1,
                                        l2,
                                        l3,
                                        l4,
                                    ] * rho[ii - 1, jj - 1, l3, l1] * dk2 / (
                                        4 * pi**2
                                    ) + cm2[
                                        (i - n + 2 * nc)
                                        + (j - n + 2 * nc - 1) * nc
                                        - 1,
                                        (ii - n + 2 * nc)
                                        + (jj - n + 2 * nc - 1) * nc
                                        - 1,
                                        l3,
                                        l2,
                                        l1,
                                        l4,
                                    ] * rho[
                                        ii - 1,
                                        jj - 1,
                                        l1,
                                        l3,
                                    ] * dk2 / (
                                        4 * pi**2
                                    )

    return hfse


# @jit(nopython=True, parallel=True)
# def hamiltonian_coulomb(data, rho, N):
#     (
#         coulomb_const,
#         num_kcutoff,
#         mapp,
#         grid,
#         C,
#         check_valid_cutoff_ar,
#     ) = data
#     ########## Note:
#     hsfe = np.zeros((N, N, 6, 6), dtype=np.complex128)
#     for k_i in nb.prange(num_kcutoff):
#         for nu in [0, 1]:
#             for alpha in [2, 3, 4, 5]:
#                 sum1 = 0.0j
#                 for k_j in range(num_kcutoff):
#                     # state1 = check_valid_cutoff_ar[mapp[k_i, 0], mapp[k_i, 1], 0]
#                     # state2 = check_valid_cutoff_ar[mapp[k_j, 0], mapp[k_j, 1], 0]
#                     #
#                     # state3 = check_valid_cutoff_ar[mapp[k_i, 0], mapp[k_i, 1], 1]
#                     # state4 = check_valid_cutoff_ar[mapp[k_j, 0], mapp[k_j, 1], 1]
#
#                     # if not (state1 and state2 or state3 and state4):
#                     # continue
#                     if k_i == k_j:
#                         continue
#                     kxi = grid[mapp[k_i, 0], mapp[k_i, 1], 0]
#                     kxj = grid[mapp[k_j, 0], mapp[k_j, 1], 0]
#                     kyi = grid[mapp[k_i, 0], mapp[k_i, 1], 1]
#                     kyj = grid[mapp[k_j, 0], mapp[k_j, 1], 1]
#                     q = 1.0 / (np.sqrt((kxj - kxi) ** 2 + (kyj - kyi) ** 2))
#
#                     for beta in [0, 1]:
#                         for mu in [2, 3, 4, 5]:
#                             C23 = C[k_i, k_j, nu, beta]
#                             C14 = C[k_j, k_i, mu, alpha]
#                             rho13 = rho[mapp[k_j, 0], mapp[k_j, 1], beta, mu]
#                             sum1 += (C14 * rho13 * C23) * q
#                 hsfe[mapp[k_i, 0], mapp[k_i, 1], nu, alpha] = sum1 * coulomb_const
#                 hsfe[mapp[k_i, 0], mapp[k_i, 1], alpha, nu] = np.conjugate(
#                     hsfe[mapp[k_i, 0], mapp[k_i, 1], nu, alpha]
#                 )
#
#     return hsfe


def CoulMat(alattice: str, wfc, grid, dk2):
    N = Config.N
    epsilon = Config.varepsilon
    epsilon0 = Config.varepsilon0
    e_charge = Config.e_charge

    # grid is an array contain kx and ky with N,N,2 dimensions
    # dk (1/nm)

    ratio = dk2 / (4 * pi**2)
    cc = e_charge**2 / (2 * epsilon * epsilon0) * ratio  # ev/nm
    print("Coulomb constant: ", cc, "ev*nm")
    # print("Ratio:", ratio)

    nkc = int(2 * N / 18)
    print("Number k cutoff:", nkc)
    cm1 = np.zeros([nkc**2, nkc**2, 6, 6, 6, 6], dtype="complex")
    cm2 = np.zeros([nkc**2, nkc**2, 6, 6, 6, 6], dtype="complex")

    with open("./results/kgridcutoff1.dat", "w", newline="") as f1:
        header = ["kx", "ky"]
        w1 = csv.DictWriter(f1, fieldnames=header, delimiter=",")
        w1.writeheader()
        for j in tqdm(range(nkc + 1, 2 * nkc + 1), desc="CM1"):
            for i in range(nkc + 1, 2 * nkc + 1):
                k = (i - nkc) + (j - nkc - 1) * nkc

                ##### Write file for the K points cutoffed zone
                rows1 = {}
                rows1["kx"] = grid[i, j, 0]
                rows1["ky"] = grid[i, j, 1]
                w1.writerow(rows1)
                for jj in range(nkc + 1, 2 * nkc + 1):
                    for ii in range(nkc + 1, 2 * nkc + 1):
                        kk = (ii - nkc) + (jj - nkc - 1) * nkc
                        q1 = sqrt(
                            (grid[i, j, 0] - grid[ii, jj, 0]) ** 2
                            + (grid[i, j, 1] - grid[ii, jj, 1]) ** 2
                        )
                        # q1 = norm(grid[i, j] - grid[ii, jj])
                        if q1 > 0:
                            for l1 in [0, 1, 2, 3]:
                                for l2 in [0, 1, 2, 3]:
                                    for l3 in [0, 1, 2, 3]:
                                        for l4 in [0, 1, 2, 3]:
                                            s11 = 0j
                                            s21 = 0j
                                            for lb in range(0, 6):
                                                s11 += (
                                                    np.conjugate(wfc[ii, jj, lb, l1])
                                                    * wfc[i, j, lb, l4]
                                                )
                                                s21 += (
                                                    np.conjugate(wfc[i, j, lb, l2])
                                                    * wfc[ii, jj, lb, l3]
                                                )
                                            cm1[k - 1, kk - 1, l1, l2, l3, l4] = (
                                                cc / q1 * s11 * s21
                                            )

    print(cm1)

    with open("./results/kgridcutoff2.dat", "w", newline="") as f2:
        header = ["kx", "ky"]
        w2 = csv.DictWriter(f2, fieldnames=header, delimiter=",")
        w2.writeheader()
        for j in tqdm(range(N - 2 * nkc + 1, N - nkc + 1), desc="CM2"):
            for i in range(N - 2 * nkc + 1, N - nkc + 1):
                k = (i - N + 2 * nkc) + (j - N + 2 * nkc - 1) * nkc

                ##### Write file for the K points cutoffed zone
                rows2 = {}
                rows2["kx"] = grid[i, j, 0]
                rows2["ky"] = grid[i, j, 1]
                w2.writerow(rows2)
                for jj in range(N - 2 * nkc + 1, N - nkc + 1):
                    for ii in range(N - 2 * nkc + 1, N - nkc + 1):
                        kk = (ii - N + 2 * nkc) + (jj - N + 2 * nkc - 1) * nkc
                        q2 = sqrt(
                            (grid[i, j, 0] - grid[ii, jj, 0]) ** 2
                            + (grid[i, j, 1] - grid[ii, jj, 1]) ** 2
                        )
                        # q2 = norm(grid[i, j] - grid[ii, jj])
                        if q2 > 0:
                            for l1 in [0, 1, 2, 3]:
                                for l2 in [0, 1, 2, 3]:
                                    for l3 in [0, 1, 2, 3]:
                                        for l4 in [0, 1, 2, 3]:
                                            s12 = 0j
                                            s22 = 0j
                                            for lb in range(0, 6):
                                                s12 += (
                                                    np.conjugate(wfc[ii, jj, lb, l1])
                                                    * wfc[i, j, lb, l4]
                                                )
                                                s22 += (
                                                    np.conjugate(wfc[i, j, lb, l2])
                                                    * wfc[ii, jj, lb, l3]
                                                )
                                            cm2[k - 1, kk - 1, l1, l2, l3, l4] = (
                                                cc / q2 * s12 * s22
                                            )

    return cm1, cm2, nkc


# def Coulomb(alattice: str, eigenVectors, grid, dk2):
#     N = Config.N
#     epsilon = Config.varepsilon
#     epsilon0 = Config.varepsilon0
#     e_charge = Config.e_charge
#
#     # grid is an array contain kx and ky with N,N,2 dimensions
#     # dk (1/nm)
#
#     ratio = dk2 / (4 * pi**2)
#     coulomb_const = e_charge**2 / (2 * epsilon * epsilon0) * ratio  # ev/nm
#     print("Coulomb constant: ", coulomb_const, "ev*nm")
#     print("Ratio:", ratio)
#
#     print("Grid before cutting off:", grid.shape)
#     num_kcutoff, check_valid_cutoff_array = cutoff_Grid_function(N, grid, alattice)
#
#     mappingArray = []
#     for i in range(N):
#         for j in range(N):
#             if check_valid_cutoff_array[i, j, 0]:
#                 mappingArray.append((i, j))
#
#     for i in range(N):
#         for j in range(N):
#             if check_valid_cutoff_array[i, j, 1]:
#                 mappingArray.append((i, j))
#
#     mappingArray = np.array(mappingArray)
#
#     if len(mappingArray) != num_kcutoff:
#         raise ValueError("Error: mapping k to map array mismatch")
#
#     # try:
#     #     indices_k1_tuple = np.where(check_valid_cutoff_array[:, :, 0])
#     #     # check True elements and return index
#     #
#     #     map_k1 = np.column_stack(indices_k1_tuple)
#     #     num_k1_points = len(map_k1)
#     #     map_k1 = np.column_stack(indices_k1_tuple)
#     #     indices_k2_tuple = np.where(check_valid_cutoff_array[:, :, 1])
#     #     map_k2 = np.column_stack(indices_k2_tuple)
#     #     mappingArray = np.vstack((map_k1, map_k2))
#     #     if len(mappingArray) != num_kcutoff:
#     #         raise ValueError(f"Error Map: ({len(mappingArray)}) and ({num_kcutoff})")
#     #
#     #     print(f"Mapping done: {mappingArray.shape}")
#     # except ValueError as e:
#     #     print(e)
#     #     import sys
#     #
#     #     sys.exit(1)
#
#     kArray = grid[mappingArray[:, 0], mappingArray[:, 1], :]
#     np.savetxt("./results/kgridcutoff.dat", kArray, delimiter=",", header="kx,ky")
#     coulombInteractionArray = np.zeros([num_kcutoff, num_kcutoff, 6, 6], dtype=complex)
#
#     for i in tqdm(range(num_kcutoff), desc="Calculating overlaps"):
#         for j in range(num_kcutoff):
#             # state1 = check_valid_cutoff_array[mappingArray[i, 0], mappingArray[i, 1], 0]
#             # state2 = check_valid_cutoff_array[mappingArray[j, 0], mappingArray[j, 1], 0]
#             #
#             # state3 = check_valid_cutoff_array[mappingArray[i, 0], mappingArray[i, 1], 1]
#             # state4 = check_valid_cutoff_array[mappingArray[j, 0], mappingArray[j, 1], 1]
#             # if state1 and state2 or state3 and state4:
#             cj = eigenVectors[mappingArray[j, 0], mappingArray[j, 1], :, :]
#             ci = eigenVectors[mappingArray[i, 0], mappingArray[i, 1], :, :]
#             S = np.conjugate(cj).T @ ci
#             # for m in range(6):
#             #     for n in range(6):
#             #         coulombInteractionArray[i, j, m, n] = S[m, n]
#             coulombInteractionArray[i, j, :, :] = S[:, :]
#
#     # print(coulombInteractionArray)
#
#     exch = (
#         coulomb_const,
#         num_kcutoff,
#         mappingArray,
#         # kArray,
#         grid,
#         coulombInteractionArray,
#         check_valid_cutoff_array,
#     )
#     return exch


def dipole(E, p, check_degenaracy):
    N = Config.N
    hbar = Config.hbar
    me = Config.m_e
    px, py = p
    xi = np.zeros([N, N, 6, 6, 2], dtype="complex")

    const = -1j * hbar / me
    for m in tqdm(range(6), desc="Dipole"):
        for n in range(6):
            if check_degenaracy[m, n] == 0:
                xi[:, :, m, n, 0] = const * (px[:, :, m, n] / (E[:, :, m] - E[:, :, n]))
                xi[:, :, m, n, 1] = const * (py[:, :, m, n] / (E[:, :, m] - E[:, :, n]))

                xi[:, :, n, m, 0] = np.conjugate(xi[:, :, m, n, 0])
                xi[:, :, n, m, 1] = np.conjugate(xi[:, :, m, n, 1])

    # return xi_x, xi_y
    return xi


def cutoff_Grid_function(N, grid, alattice) -> Tuple[int, NDArray[np.bool_]]:
    """
    Apply a momentum-space cutoff around the K and K' points in the Brillouin zone.

    This function scans through all k-points in a 2D reciprocal-space grid and determines
    whether each point lies within a cutoff radius `rkcutoff` around either of the two
    Dirac points K and K'. It returns both the total count of valid points and a boolean
    mask identifying which grid points belong to which valley.

    Parameters
    ----------
    N : int
        The number of k-points along each dimension of the Monkhorst–Pack grid.

    grid : ndarray of shape (N, N, 2)
        The array of k-points in reciprocal space.
        Each element `grid[i, j] = [kx, ky]` gives the Cartesian components
        of the k-point in units of 1/length.

    alattice : float
        The lattice constant of the real-space unit cell.
        Used to locate the K and K' points at ±4π/(3a) along the kx-axis.

    Returns
    -------
    count : int
        The total number of k-points lying within the cutoff radius of either
        K or K'.

    check_valid_cutoff_grid : ndarray of shape (N, N, 2), bool
        A boolean mask indicating whether each k-point belongs to:
        - `[..., 0]`: inside the cutoff around the K valley (+4π/3a)
        - `[..., 1]`: inside the cutoff around the K' valley (−4π/3a)
        Points outside both regions are False.

    Notes
    -----
    - The cutoff radius `rkcutoff` is taken from `Config.rkcutoff`.
    - The K and K' points are defined as:
        K  = ( +4π / (3a), 0 )
        K' = ( −4π / (3a), 0 )
    - This function is typically used to limit calculations (e.g., Coulomb integrals)
      to regions around the Dirac points, reducing computational cost.

    Example
    -------
    >>> count, mask = cutoff_Grid_function(N=301, grid=kArray, alattice=2.46)
    >>> print(count)
    520
    >>> mask.shape
    (301, 301, 2)
    >>> mask[150, 150, 0]
    False
    >>> mask[120, 180, 1]
    True
    """
    rkcutoff = Config.rkcutoff
    print("radius cutoff k:", rkcutoff)
    check_valid_cutoff_grid = np.zeros([N, N, 2], dtype=bool)
    count = 0
    for i in tqdm(range(N), desc="Cutting off K-points"):
        for j in range(N):
            rk1 = sqrt(
                (grid[i, j, 0] - 4 * pi / (3 * alattice)) ** 2 + grid[i, j, 1] ** 2
            )
            rk2 = sqrt(
                (grid[i, j, 0] + 4 * pi / (3 * alattice)) ** 2 + grid[i, j, 1] ** 2
            )
            if rk1 <= rkcutoff:
                check_valid_cutoff_grid[i, j, 0] = True
                count += 1
            elif rk2 <= rkcutoff:
                check_valid_cutoff_grid[i, j, 1] = True
                count += 1
    print("Number of k points after cutting off:", count)
    return count, check_valid_cutoff_grid


def solveHamiltonian(data, kArray, modelNeighbor) -> Tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    N = Config.N
    alattice = data["alattice"]
    me = Config.m_e
    hbar = Config.hbar
    # grid K phai chay lai moi lan do khac alattice

    eigenValues = np.zeros([N, N, 6])

    eigenVectors = np.zeros([N, N, 6, 6], dtype="complex")
    pmx = np.zeros([N, N, 6, 6], dtype="complex")
    pmy = np.zeros([N, N, 6, 6], dtype="complex")

    ham_spin = np.zeros([6, 6], dtype="complex")
    dhkx_spin = np.zeros([6, 6], dtype="complex")
    dhky_spin = np.zeros([6, 6], dtype="complex")
    for j in tqdm(range(N), desc="Calculate bandstructure"):
        for i in range(N):
            alpha = kArray[i, j, 0] / 2 * alattice
            beta = sqrt(3) / 2 * kArray[i, j, 1] * alattice
            ham, dhkx, dhky, hamu, hamd = tbm_Hamiltonian(
                alpha, beta, data, modelNeighbor, alattice
            )
            # vals, vecs = np.linalg.eigh(ham)
            ############ co xet spin
            ham_spin[0:3, 0:3] = hamu
            ham_spin[3:6, 3:6] = hamd

            dhkx_spin[0:3, 0:3] = dhkx
            dhkx_spin[3:6, 3:6] = dhkx

            dhky_spin[0:3, 0:3] = dhky
            dhky_spin[3:6, 3:6] = dhky

            vals, vecs = np.linalg.eigh(ham_spin)

            eigenValues[i, j, :] = vals
            eigenVectors[i, j, :, :] = vecs

            pmx_matrix = np.conjugate(vecs).T @ dhkx_spin @ vecs
            pmy_matrix = np.conjugate(vecs).T @ dhky_spin @ vecs

            pmx[i, j, :, :] = pmx_matrix * me / hbar
            pmy[i, j, :, :] = pmy_matrix * me / hbar
            ########################################### Code o tren tuong duong code o duoi
            # for jb in range(0, 6):
            #     eigenValues[i, j, jb] = vals[jb]
            #     for ib in range(0, 6):
            #         sum1 = 0j
            #         sum2 = 0j
            #         for jjb in range(0, 6):
            #             for iib in range(0, 6):
            #                 sum1 += np.conjugate(vecs[iib, ib]) * dhkx_spin[iib, jjb] * vecs[jjb, jb]
            #                 sum2 += np.conjugate(vecs[iib, ib]) * dhky_spin[iib, jjb] * vecs[jjb, jb]
            #         pmx[i, j, ib, jb] = sum1 * me / hbar
            #         pmy[i, j, ib, jb] = sum2 * me / hbar

    return eigenValues, eigenVectors, pmx, pmy
