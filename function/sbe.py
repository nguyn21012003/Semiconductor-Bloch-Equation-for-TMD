import csv
from typing import Tuple

import numba as nb
import numpy as np
import pandas as pd
from numba import jit
from numpy import pi, sqrt
from numpy.typing import NDArray
from tqdm import tqdm

from core import genGrid, pulse
from core.genHam import tbm_Hamiltonian
from settings.configs import Config


def linearSBE(data, modelNeighbor):
    alattice = data["alattice"]
    N = Config.N
    detuning = Config.detuning

    # grid, dkx, dky = genGrid.Rhombus(alattice, N)
    grid, dk2 = genGrid.Monkhorst(alattice, N)
    # grid, dk2 = genGrid.Cartesian(alattice, N)
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
    for jb in range(2):
        for ib in range(2, 6):
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

    # hfse = HartreeFockSelfEnergy(rho, dk2, N, nkc, cm1, cm2)
    for v in range(2):
        for c in range(2, 6):
            # ω_cv = (E_c − E_v)/ħ
            omegaCV = (E[:, :, c] - E[:, :, v]) / Config.hbar

            # dipole element ξ_cv
            xix = xiov[:, :, c, v, 0]
            xiy = xiov[:, :, c, v, 1]
            omegaRabi = (Etx * xix + Ety * xiy) / Config.hbar

            # HF term for (c,v)
            # omegaHF = hfse[:, :, c, v] / Config.hbar

            # density matrix element ρ_cv
            rho_cv = rho[:, :, c, v]

            # equation of motion for ρ_cv
            rhodot = (
                -1j * (omegaCV - omega_L) * rho_cv
                # + 1j * omegaHF
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
            for l4 in range(4):
                for l2 in range(4):
                    for jj in range(nc + 1, 2 * nc + 1):
                        for ii in range(nc + 1, 2 * nc + 1):
                            for l1 in range(2):
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
                            for l3 in range(2, 4):
                                hfse[i - 1, j - 1, l2, l4] += (
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
                            for l1 in range(3):
                                for l3 in range(l1 + 1, 4):
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
                                        ii - 1, jj - 1, l1, l3
                                    ] * dk2 / (
                                        4 * pi**2
                                    )

    for j in nb.prange(n - 2 * nc + 1, n - nc + 1):
        for i in range(n - 2 * nc + 1, n - nc + 1):
            for l4 in range(4):
                for l2 in range(4):
                    for jj in range(n - 2 * nc + 1, n - nc + 1):
                        for ii in range(n - 2 * nc + 1, n - nc + 1):
                            for l1 in range(2):
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
                            for l3 in range(2, 4):
                                hfse[i - 1, j - 1, l2, l4] += (
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
                            for l1 in range(3):
                                for l3 in range(l1 + 1, 4):
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
                                        ii - 1, jj - 1, l1, l3
                                    ] * dk2 / (
                                        4 * pi**2
                                    )

    return hfse


@jit(nopython=True, parallel=True)
def calcCMp(n, nc, cc, wfc, grid):
    cm1 = np.zeros((nc**2, nc**2, 4, 4, 4, 4), dtype=np.complex128)
    cm2 = np.zeros((nc**2, nc**2, 4, 4, 4, 4), dtype=np.complex128)
    for j in nb.prange(nc + 1, 2 * nc + 1):
        for i in range(nc + 1, 2 * nc + 1):
            k1 = (i - nc) + (j - nc - 1) * nc
            for jj in range(nc + 1, 2 * nc + 1):
                for ii in range(nc + 1, 2 * nc + 1):
                    kk = (ii - nc) + (jj - nc - 1) * nc
                    # q1 = sqrt(
                    #     (grid[i, j, 0] - grid[ii, jj, 0]) ** 2
                    #     + (grid[i, j, 1] - grid[ii, jj, 1]) ** 2
                    # )
                    q1 = np.linalg.norm(grid[i, j] - grid[ii, jj])
                    if q1 > 0:
                        for l1 in range(4):
                            for l2 in range(4):
                                for l3 in range(4):
                                    for l4 in range(4):
                                        s11 = 0j
                                        s21 = 0j
                                        for lb in range(6):
                                            s11 += (
                                                np.conjugate(wfc[ii, jj, lb, l1])
                                                * wfc[i, j, lb, l4]
                                            )
                                            s21 += (
                                                np.conjugate(wfc[i, j, lb, l2])
                                                * wfc[ii, jj, lb, l3]
                                            )
                                        cm1[k1 - 1, kk - 1, l1, l2, l3, l4] = (
                                            cc / q1 * s11 * s21
                                        )

    for j in nb.prange(n - 2 * nc + 1, n - nc + 1):
        for i in range(n - 2 * nc + 1, n - nc + 1):
            k = (i - n + 2 * nc) + (j - n + 2 * nc - 1) * nc
            for jj in range(n - 2 * nc + 1, n - nc + 1):
                for ii in range(n - 2 * nc + 1, n - nc + 1):
                    kk = (ii - n + 2 * nc) + (jj - n + 2 * nc - 1) * nc
                    # q2 = sqrt(
                    #     (grid[i, j, 0] - grid[ii, jj, 0]) ** 2
                    #     + (grid[i, j, 1] - grid[ii, jj, 1]) ** 2
                    # )
                    q2 = np.linalg.norm(grid[i, j] - grid[ii, jj])
                    if q2 > 0:
                        for l1 in range(4):
                            for l2 in range(4):
                                for l3 in range(4):
                                    for l4 in range(4):
                                        s12 = 0j
                                        s22 = 0j
                                        for lb in range(6):
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

    return cm1, cm2


def CoulMat(alattice: str, wfc, grid, dk2):
    n = Config.N
    epsilon = Config.varepsilon
    print("epsilon: ", epsilon)
    epsilon0 = Config.varepsilon0
    print("epsilon0: ", epsilon0)
    e_charge = Config.e_charge

    # grid is an array contain kx and ky with N,N,2 dimensions

    cc = 2 * pi * e_charge**2 / (4 * pi * epsilon * epsilon0)  # ev/nm
    print("Coulomb constant: ", cc, "ev*nm")
    # print("Ratio:", ratio)

    nc = int(2 * n / 18)
    print("Number k cutoff:", nc)

    cm1, cm2 = calcCMp(n, nc, cc, wfc, grid)
    with open("./results/kgridcutoff1.dat", "w", newline="") as f1:
        header = ["kx", "ky"]
        w1 = csv.DictWriter(f1, fieldnames=header, delimiter=",")
        w1.writeheader()
        for j in tqdm(range(nc + 1, 2 * nc + 1), desc="CM1"):
            for i in range(nc + 1, 2 * nc + 1):

                ##### Write file for the K points cutoffed zone
                rows1 = {}
                rows1["kx"] = grid[i - 1, j - 1, 0]
                rows1["ky"] = grid[i - 1, j - 1, 1]
                w1.writerow(rows1)

    with open("./results/kgridcutoff2.dat", "w", newline="") as f2:
        header = ["kx", "ky"]
        w2 = csv.DictWriter(f2, fieldnames=header, delimiter=",")
        w2.writeheader()
        for j in tqdm(range(n - 2 * nc + 1, n - nc + 1), desc="CM2"):
            for i in range(n - 2 * nc + 1, n - nc + 1):

                ##### Write file for the K points cutoffed zone
                rows2 = {}
                rows2["kx"] = grid[i - 1, j - 1, 0]
                rows2["ky"] = grid[i - 1, j - 1, 1]
                w2.writerow(rows2)

    return cm1, cm2, nc


def dipole(E, p, check_degenaracy):
    N = Config.N
    hbar = Config.hbar
    me = Config.m_e
    px, py = p
    xi = np.zeros([N, N, 6, 6, 2], dtype="complex")

    const = -1j * hbar / me
    for m in tqdm(range(6), desc="Dipole"):
        for n in range(m + 1, 6):
            if check_degenaracy[m, n] == 0:
                if m != n:
                    xi[:, :, m, n, 0] = const * (
                        px[:, :, m, n] / (E[:, :, m] - E[:, :, n])
                    )
                    xi[:, :, m, n, 1] = const * (
                        py[:, :, m, n] / (E[:, :, m] - E[:, :, n])
                    )

                    xi[:, :, n, m, 0] = np.conjugate(xi[:, :, m, n, 0])
                    xi[:, :, n, m, 1] = np.conjugate(xi[:, :, m, n, 1])

    return xi


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
