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
    grid, dk2 = genGrid.Monkhorst0(alattice, N)
    # grid, dk2 = genGrid.Cartesian(alattice, N)
    print(dk2)
    print(f"Grid dim:{grid.shape}")

    plot_grid(grid, "kgrid.dat")

    eigenValues, eigenVectors, pmx, pmy = solveHamiltonian(data, grid, modelNeighbor)

    egap = np.min(eigenValues[:, :, 2]) - np.max(eigenValues[:, :, 1])
    print(f"Energy gap: {egap} eV")
    print(f"Detuning: {detuning} eV")

    w0 = (egap + detuning) / Config.hbar
    print(f"Frequency: {w0}")

    check_degenaracy = np.zeros([6, 6])
    for i in tqdm(range(N), desc="Check degenaracy"):
        for j in range(N):
            for nu in range(6):
                for mu in range(nu + 1, 6):
                    diff = np.abs(eigenValues[i, j, nu] - eigenValues[i, j, mu])
                    if diff < 1e-7:
                        check_degenaracy[mu, nu] = 1
                        check_degenaracy[nu, mu] = 1

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
    print("Pulse width: ", tL)
    E0 = Config.E0
    ntmax = int((tmax - tmin) / dt)

    # data_Coulomb = Coulomb(alattice, eigenVectors, grid, dk2)
    CM = CoulMat(eigenVectors, grid, dk2, alattice)
    # data_Coulomb = None
    ############################################################ Calculating SBE
    P_t_array = np.zeros([ntmax, 2], dtype="complex")  # Save P(t)
    density = np.zeros([ntmax, 2])
    time_array = np.zeros(ntmax, dtype=float)
    Atx, Aty = 0, 0
    t = tmin

    writePulse(ntmax, tmin, dt, E0, tL, w0)

    for nt in tqdm(range(ntmax), desc="RK4 SBE: ", colour="blue"):
        time = t + nt * dt
        Etx, Ety = pulse.Et(time, E0, tL, w0)
        Etime = (Etx, Ety)
        Atx = Config.e_charge / Config.m_e * (Atx - Etx * dt)
        Aty = Config.e_charge / Config.m_e * (Aty - Ety * dt)
        Atime = (Atx, Aty)

        # rho = rk4(rho, t, dt, Etime, Atime, xi, p, dk2, data_Coulomb, eigenValues, w0)
        rho = rk4(rho, time, dt, Etime, Atime, xi, p, CM, eigenValues, w0, dk2)
        Pxt, Pyt = polarization(rho, xi, dk2)
        ne, nh = calcdensity(rho, dk2)

        P_t_array[nt, 0] = Pxt
        P_t_array[nt, 1] = Pyt

        density[nt, 0] = np.real(ne)
        density[nt, 1] = np.real(nh)
        time_array[nt] = time

    print(time_array)
    absorptionSpectra1(P_t_array, egap, time_array, w0)
    distributionCarriers(density, time_array)


def writePulse(ntmax, tmin, dt, E0, tL, w0):
    with open("./results/pulse/pulse.dat", "w") as f:
        header = ["time", "Ex", "Ey"]
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for nt in tqdm(range(1, ntmax + 1), desc="Pulse"):
            time = tmin + nt * dt
            Ext, Eyt = pulse.Et(time, E0, tL, w0)
            row = {"time": time, "Ex": Ext, "Ey": Eyt}
            writer.writerow(row)
    return None


def distributionCarriers(dens, evoltime):
    output_file = "./results/distribution.csv"

    header = ["t", "ne", "nh"]
    data = {
        "t": evoltime,
        "ne": dens[:, 0],
        "nh": dens[:, 1],
    }
    df = pd.DataFrame(data, columns=header)
    df.to_csv(output_file, index=False)


def calcdensity(rho, dk2):
    ne = 0.0
    nh = 0.0
    for mu in range(2):
        nh += dk2 / (4 * pi**2) * rho[:, :, mu, mu]
    for nu in range(2, 6):
        ne += dk2 / (4 * pi**2) * (1 - rho[:, :, nu, nu])
    ne = ne.sum()
    nh = nh.sum()

    return ne, nh


def absorptionSpectra(Pt, egap, evoltime, omegaL):
    hbar = Config.hbar
    dt = Config.dt
    E0 = Config.E0
    Nw = 2000
    tL = Config.time_duration
    output_file = "./results/AbSpect.csv"

    omega_start = (egap - 2.0) / hbar
    omega_end = (egap + 1.0) / hbar
    domega = (omega_end - omega_start) / Nw

    energy_minus_egap = []
    chi_x_real, chi_x_imag = [], []
    chi_y_real, chi_y_imag = [], []
    ew_real, ew_imag = [], []
    alpha_w = []

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

    omega = omega_start
    pbar = tqdm(total=int((omega_end - omega_start) / domega), desc="linearAbsorpt")

    while omega <= omega_end:
        s1 = 0j
        s2 = 0j
        s3 = 0j

        for it in range(len(evoltime)):
            t = evoltime[it]
            phase = np.exp(1j * (omega - omegaL) * t)

            s1 += Pt[it, 0] * phase * dt
            s2 += Pt[it, 1] * phase * dt

            Et_val = pulse.Et(t, E0, tL, omegaL)
            s3 += Et_val[0] * phase * dt
        denom = s3 if abs(s3) > 1e-20 else 1e-20
        chi_x = s1 / denom
        chi_y = s2 / denom

        energy_minus_egap.append(omega * hbar - egap)
        chi_x_real.append(chi_x.real)
        chi_x_imag.append(chi_x.imag)
        chi_y_real.append(chi_y.real)
        chi_y_imag.append(chi_y.imag)
        ew_real.append(s3.real)
        ew_imag.append(s3.imag)

        alpha_w.append(chi_x.imag + chi_y.imag)

        omega += domega
        pbar.update(1)

    pbar.close()

    data = {
        "Energy_minus_Egap(eV)": energy_minus_egap,
        "Re(Chi_x)": chi_x_real,
        "Im(Chi_x)": chi_x_imag,
        "Re(Chi_y)": chi_y_real,
        "Im(Chi_y)": chi_y_imag,
        "Re(E_w)": ew_real,
        "Im(E_w)": ew_imag,
        "Alpha(w)": alpha_w,
    }

    df = pd.DataFrame(data, columns=header)
    df.to_csv(output_file, index=False, float_format="%.10g")
    print(f"file: {output_file}")

    return True


def absorptionSpectra1(P_t_array, egap, evoltime, wL):
    hbar = Config.hbar
    dt = Config.dt
    E0 = Config.E0
    tL = Config.time_duration
    omega = (egap - 2) / hbar
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
    Ex_array = np.array([pulse.Et(t, E0, tL, wL)[0] for t in evoltime])
    P_t_x = P_t_array[:, 0]
    P_t_y = P_t_array[:, 1]
    delta_omega_col = (omega_array - wL)[:, np.newaxis]
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


def polarization(rho, xi, dk2):
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


def rk4(rho, t, dt, Et, At, xi, pm, exchange_term, E, w0, dk2):
    k1 = rhsSBE_LG(rho, t, Et, xi, E, w0, exchange_term, dk2)
    k2 = rhsSBE_LG(rho + 0.5 * k1 * dt, t, Et, xi, E, w0, exchange_term, dk2)
    k3 = rhsSBE_LG(rho + 0.5 * k2 * dt, t, Et, xi, E, w0, exchange_term, dk2)
    k4 = rhsSBE_LG(rho + 1.0 * k3 * dt, t, Et, xi, E, w0, exchange_term, dk2)

    # k1 = eqLG(rho, t, Et, xi, E, w0, exchange_term, dk2)
    # k2 = eqLG(rho + 0.5 * k1 * dt, t, Et, xi, E, w0, exchange_term, dk2)
    # k3 = eqLG(rho + 0.5 * k2 * dt, t, Et, xi, E, w0, exchange_term, dk2)
    # k4 = eqLG(rho + 1.0 * k3 * dt, t, Et, xi, E, w0, exchange_term, dk2)

    # k1 = rhsSBE_VG(rho, t, Et, At, E, pm, w0, exchange_term, dk2)
    # k2 = rhsSBE_VG(rho + 0.5 * k1 * dt, t, Et, At, E, pm, w0, exchange_term, dk2)
    # k3 = rhsSBE_VG(rho + 0.5 * k2 * dt, t, Et, At, E, pm, w0, exchange_term, dk2)
    # k4 = rhsSBE_VG(rho + 1.0 * k3 * dt, t, Et, At, E, pm, w0, exchange_term, dk2)
    return rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rhsSBE_VG(rho, t, Et, At, E, p, omega_L, exch_term):
    ta = Config.tmin
    dt = Config.dt
    qe = Config.e_charge
    x1 = int((t - ta) / dt + 0.1)
    x2 = round((t - ta) / dt + 0.1)
    Etx, Ety = Et
    Atx, Aty = At

    N = Config.N
    Y = np.zeros((N, N, 6, 6), dtype="complex")
    T2 = Config.T_cohenrent

    px, py = p
    cm1, cm2, nkc = exch_term
    # hfse = HartreeFockSelfEnergy(rho, N, nkc, cm1, cm2)
    if abs(x1 - x2) == 0:
        eAx = qe * (Atx - Etx * dt)
        eAy = qe * (Aty - Ety * dt)
    else:
        eAx = qe * (Atx - Etx * dt / 2)
        eAy = qe * (Aty - Ety * dt / 2)

    for v in range(2):
        for c in range(6):
            omegaCV = (E[:, :, c] - E[:, :, v]) / Config.hbar
            pmx = px[:, :, c, v]
            pmy = py[:, :, c, v]
            omegaRabi = (eAx * pmx + eAy * pmy) / Config.hbar
            # omegaHF = hfse[:, :, c, v] / Config.hbar
            rho_cv = rho[:, :, c, v]
            rhodot = (
                -1j * (omegaCV - omega_L) * rho_cv
                # + 1j * omegaHF
                - 1j * omegaRabi
                - rho_cv / T2
            )

            Y[:, :, c, v] = rhodot
            Y[:, :, v, c] = np.conjugate(rhodot)

    return Y


def eqLG(rho, t, Et, xiov, E, omega_L, exch_term, dk2):
    N = Config.N
    T1 = Config.T_relaxation
    T2 = Config.T_cohenrent
    qe = Config.e_charge
    hb = Config.hbar
    Etx, Ety = Et
    cm1, cm2, nkc = exch_term
    hfse = HartreeFockSelfEnergy(rho, N, nkc, cm1, cm2)

    # kappa, map1, nkc, grid = exch_term
    # hfse = HartreeFockSelfEnergy_circle(N, nkc, dk2, kappa, grid, map1, rho)
    H = np.zeros((N, N, 6, 6), dtype="complex")
    Y = np.zeros((N, N, 6, 6), dtype="complex")
    #### If use RWA
    for m in range(6):
        if m < 2:
            H[:, :, m, m] = E[:, :, m]
        else:
            H[:, :, m, m] = E[:, :, m] - omega_L * hb
    ####################################################################
    H = H - qe * (Etx * xiov[:, :, :, :, 0] + Ety * xiov[:, :, :, :, 1])
    wCoulomb = True
    if wCoulomb:
        H = H - hfse
    comm = H @ rho - rho @ H

    for m in range(6):
        for n in range(m, 6):
            # if n == m:
            #     Y[:, :, m, m] = np.imag(comm[:, :, m, n]) / hb
            #     if Config.useRelaxation:
            #         Y[:, :, m, n] -= np.real(rho[:, :, m, n]) / T1
            if n != m:
                Y[:, :, m, n] = -1j * comm[:, :, m, n] / hb - rho[:, :, m, n] / T2
                Y[:, :, n, m] = np.conjugate(Y[:, :, m, n])

    return Y


def rhsSBE_LG(rho, t, Et, xiov, E, omega_L, exch_term, dk2):
    Etx, Ety = Et
    N = Config.N
    Y = np.zeros((N, N, 6, 6), dtype="complex")
    T2 = Config.T_cohenrent
    # hfse = hamiltonian_coulomb(exchange_term, rho, N)
    # cm1, cm2, nkc = exch_term
    # hfse = HartreeFockSelfEnergy(rho, N, nkc, cm1, cm2)

    cm1, cm2, mapk1, mapk2 = exch_term
    hfse = HartreeFockSelfEnergy_circle(rho, N, mapk1, mapk2, cm1, cm2)

    # kappa, map1, nkc, grid = exch_term
    # hfse = HartreeFockSelfEnergy_circle(N, nkc, dk2, kappa, grid, map1, rho)
    for v in range(2):
        for c in range(2, 6):
            omegaCV = (E[:, :, c] - E[:, :, v]) / Config.hbar
            xix = xiov[:, :, c, v, 0]
            xiy = xiov[:, :, c, v, 1]
            omegaRabi = (Etx * xix + Ety * xiy) / Config.hbar
            omegaHF = hfse[:, :, c, v] / Config.hbar
            rho_cv = rho[:, :, c, v]
            rhodot = (
                -1j * (omegaCV - omega_L) * rho_cv
                + 1j * omegaHF
                - 1j * omegaRabi
                - rho_cv / T2
            )

            Y[:, :, c, v] = rhodot
            Y[:, :, v, c] = np.conjugate(rhodot)

    return Y


def plot_grid(kArr, file):
    kx = kArr[:, :, 0].flatten()
    ky = kArr[:, :, 1].flatten()
    np.savetxt(
        "./results/" + file, np.column_stack((kx, ky)), header="kx,ky", delimiter=","
    )


# @jit(nopython=True, parallel=True)
# def hsfep(nk, nc, dk2, kappa, grid, map1, rho, eps, eps0):
#     hfse = np.zeros((nk, nk, 6, 6), dtype=np.complex128)
#     coeff1 = 1 / (2 * eps0) * dk2 / (4 * pi**2)
#     for j in nb.prange(nc):
#         for m in range(4):
#             for n in range(m, 4):
#                 s1 = 0
#                 for i in range(nc):
#                     if i != j:
#                         ki_x = grid[map1[i, 0], map1[i, 1], 0]
#                         ki_y = grid[map1[i, 0], map1[i, 1], 1]
#                         kj_x = grid[map1[j, 0], map1[j, 1], 0]
#                         kj_y = grid[map1[j, 0], map1[j, 1], 1]
#
#                         q = np.sqrt((ki_x - kj_x) ** 2 + (ki_y - kj_y) ** 2)
#                         coeff2 = 1 / (q * eps)
#                         if m <= 2 and n > 2:
#                             for mm in range(2, 4):
#                                 for nn in range(3):
#                                     s1 += (
#                                         kappa[i, j, mm, n]
#                                         * kappa[j, i, m, nn]
#                                         * rho[map1[i, 0], map1[i, 1], nn, mm]
#                                         + kappa[i, j, nn, n]
#                                         * kappa[j, i, m, mm]
#                                         * rho[map1[i, 0], map1[i, 1], mm, nn]
#                                     ) * coeff2
#                         elif n == m and m <= 2:
#                             for mm in range(2):
#                                 for nn in range(2):
#                                     s1 -= (
#                                         kappa[i, j, mm, n]
#                                         * kappa[j, i, m, nn]
#                                         * (1 - rho[map1[i, 0], map1[i, 1], nn, mm])
#                                     ) * coeff2
#                         elif n == m and m > 2:
#                             for mm in range(2, 4):
#                                 for nn in range(2, 4):
#                                     s1 += (
#                                         kappa[i, j, mm, n]
#                                         * kappa[j, i, m, nn]
#                                         * rho[map1[i, 0], map1[i, 1], nn, mm]
#                                     ) * coeff2
#                 hfse[map1[j, 0], map1[j, 1], m, n] = s1 * coeff1
#                 if n != m:
#                     hfse[map1[j, 0], map1[j, 1], n, m] = np.conjugate(s1 * coeff1)
#
#     return hfse


# def HartreeFockSelfEnergy_circle(nk, nc, dk2, kappa, grid, map1, rho):
#     eps0 = Config.varepsilon0
#     eps = Config.varepsilon
#     hfse = hsfep(nk, nc, dk2, kappa, grid, map1, rho, eps, eps0)
#
#     return hfse


@jit(nopython=True, parallel=True)
def HartreeFockSelfEnergy(rho, n, nc, cm1, cm2):
    hfse = np.zeros((n, n, 6, 6), dtype=np.complex128)
    shift = 0
    start1 = nc + 1 - shift
    width = nc
    end1 = start1 + width

    for j in nb.prange(start1, end1):
        for i in range(start1, end1):
            k = (i - start1 + 1) + (j - start1) * width
            for jj in range(start1, end1):
                for ii in range(start1, end1):
                    kk = (ii - start1 + 1) + (jj - start1) * width
                    for l4 in range(4):
                        for l2 in range(4):
                            for l1 in range(2):
                                hfse[i - 1, j - 1, l2, l4] += -(
                                    cm1[k - 1, kk - 1, l1, l2, l1, l4]
                                    * (1.0 - rho[ii - 1, jj - 1, l1, l1])
                                )
                            for l3 in range(2, 4):
                                hfse[i - 1, j - 1, l2, l4] += (
                                    cm1[k - 1, kk - 1, l3, l2, l3, l4]
                                    * rho[ii - 1, jj - 1, l3, l3]
                                )
                            for l1 in range(3):
                                for l3 in range(l1 + 1, 4):
                                    hfse[i - 1, j - 1, l2, l4] += (
                                        cm1[k - 1, kk - 1, l1, l2, l3, l4]
                                        * rho[ii - 1, jj - 1, l3, l1]
                                        + cm1[k - 1, kk - 1, l3, l2, l1, l4]
                                        * rho[ii - 1, jj - 1, l1, l3]
                                    )

    start2 = n - 2 * nc + 1 + shift
    end2 = start2 + width
    for j in nb.prange(start2, end2):
        for i in range(start2, end2):
            k = (i - start2 + 1) + (j - start2) * width
            for jj in range(start2, end2):
                for ii in range(start2, end2):
                    kk = (ii - start2 + 1) + (jj - start2) * width
                    for l4 in range(4):
                        for l2 in range(4):
                            for l1 in range(2):
                                hfse[i - 1, j - 1, l2, l4] += -(
                                    cm2[k - 1, kk - 1, l1, l2, l1, l4]
                                    * (1.0 - rho[ii - 1, jj - 1, l1, l1])
                                )
                            for l3 in range(2, 4):
                                hfse[i - 1, j - 1, l2, l4] += (
                                    cm2[k - 1, kk - 1, l3, l2, l3, l4]
                                    * rho[ii - 1, jj - 1, l3, l3]
                                )
                            for l1 in range(3):
                                for l3 in range(l1 + 1, 4):
                                    hfse[i - 1, j - 1, l2, l4] += (
                                        cm2[k - 1, kk - 1, l1, l2, l3, l4]
                                        * rho[ii - 1, jj - 1, l3, l1]
                                        + cm2[k - 1, kk - 1, l3, l2, l1, l4]
                                        * rho[ii - 1, jj - 1, l1, l3]
                                    )

    return hfse


@jit(nopython=True, parallel=True)
def calcCMp(n, nc, cc, wfc, grid):
    cm1 = np.zeros((nc**2, nc**2, 4, 4, 4, 4), dtype=np.complex128)
    cm2 = np.zeros((nc**2, nc**2, 4, 4, 4, 4), dtype=np.complex128)
    shift = 0
    start1 = nc + 1 - shift
    width = nc
    end1 = start1 + width
    for j in nb.prange(start1, end1):
        for i in range(start1, end1):
            k = (i - start1 + 1) + (j - start1) * width
            for jj in range(start1, end1):
                for ii in range(start1, end1):
                    kk = (ii - start1 + 1) + (jj - start1) * width
                    # q1 = sqrt(
                    #     (grid[i, j, 0] - grid[ii, jj, 0]) ** 2
                    #     + (grid[i, j, 1] - grid[ii, jj, 1]) ** 2
                    # )
                    q1 = np.linalg.norm(grid[i - 1, j - 1] - grid[ii - 1, jj - 1])
                    if q1 > 0:
                        for l1 in range(4):
                            for l2 in range(4):
                                for l3 in range(4):
                                    for l4 in range(4):
                                        s11 = 0j
                                        s21 = 0j
                                        for lb in range(6):
                                            s11 += (
                                                np.conjugate(
                                                    wfc[ii - 1, jj - 1, lb, l1]
                                                )
                                                * wfc[i - 1, j - 1, lb, l4]
                                            )
                                            s21 += (
                                                np.conjugate(wfc[i - 1, j - 1, lb, l2])
                                                * wfc[ii - 1, jj - 1, lb, l3]
                                            )
                                        cm1[k - 1, kk - 1, l1, l2, l3, l4] = (
                                            cc / q1 * s11 * s21
                                        )

    start2 = n - 2 * nc + 1 + shift
    end2 = start2 + width
    for j in nb.prange(start2, end2):
        for i in range(start2, end2):
            k = (i - start2 + 1) + (j - start2) * width
            for jj in range(start2, end2):
                for ii in range(start2, end2):
                    kk = (ii - start2 + 1) + (jj - start2) * width
                    # q2 = sqrt(
                    #     (grid[i, j, 0] - grid[ii, jj, 0]) ** 2
                    #     + (grid[i, j, 1] - grid[ii, jj, 1]) ** 2
                    # )
                    q2 = np.linalg.norm(grid[i - 1, j - 1] - grid[ii - 1, jj - 1])
                    if q2 > 0:
                        for l1 in range(4):
                            for l2 in range(4):
                                for l3 in range(4):
                                    for l4 in range(4):
                                        s12 = 0j
                                        s22 = 0j
                                        for lb in range(6):
                                            s12 += (
                                                np.conjugate(
                                                    wfc[ii - 1, jj - 1, lb, l1]
                                                )
                                                * wfc[i - 1, j - 1, lb, l4]
                                            )
                                            s22 += (
                                                np.conjugate(wfc[i - 1, j - 1, lb, l2])
                                                * wfc[ii - 1, jj - 1, lb, l3]
                                            )
                                        cm2[k - 1, kk - 1, l1, l2, l3, l4] = (
                                            cc / q2 * s12 * s22
                                        )

    return cm1, cm2


@jit(nopython=True, parallel=True)
def HartreeFockSelfEnergy_circle(rho, n, mapk1, mapk2, cm1, cm2):
    hfse = np.zeros((n, n, 6, 6), dtype=np.complex128)
    nk1 = len(mapk1)
    nk2 = len(mapk2)

    for k in nb.prange(nk1):
        i = mapk1[k, 0]
        j = mapk1[k, 1]
        for kk in range(nk1):
            ii = mapk1[kk, 0]
            jj = mapk1[kk, 1]
            for l4 in range(4):
                for l2 in range(4):
                    for l1 in range(2):
                        hfse[i, j, l2, l4] += -(
                            cm1[k, kk, l1, l2, l1, l4] * (1.0 - rho[ii, jj, l1, l1])
                        )
                    for l3 in range(2, 4):
                        hfse[i, j, l2, l4] += (
                            cm1[k, kk, l3, l2, l3, l4] * rho[ii, jj, l3, l3]
                        )
                    for l1 in range(3):
                        for l3 in range(l1 + 1, 4):
                            hfse[i, j, l2, l4] += (
                                cm1[k, kk, l1, l2, l3, l4] * rho[ii, jj, l3, l1]
                                + cm1[k, kk, l3, l2, l1, l4] * rho[ii, jj, l1, l3]
                            )

    for k in nb.prange(nk2):
        i = mapk2[k, 0]
        j = mapk2[k, 1]
        for kk in range(nk2):
            ii = mapk2[kk, 0]
            jj = mapk2[kk, 1]
            for l4 in range(4):
                for l2 in range(4):
                    for l1 in range(2):
                        hfse[i, j, l2, l4] += -(
                            cm2[k, kk, l1, l2, l1, l4] * (1.0 - rho[ii, jj, l1, l1])
                        )
                    for l3 in range(2, 4):
                        hfse[i, j, l2, l4] += (
                            cm2[k, kk, l3, l2, l3, l4] * rho[ii, jj, l3, l3]
                        )
                    for l1 in range(3):
                        for l3 in range(l1 + 1, 4):
                            hfse[i, j, l2, l4] += (
                                cm2[k, kk, l1, l2, l3, l4] * rho[ii, jj, l3, l1]
                                + cm2[k, kk, l3, l2, l1, l4] * rho[ii, jj, l1, l3]
                            )

    return hfse


@jit(nopython=True, parallel=True)
def calcCMp_circle(mapk1, mapk2, cc, wfc, grid):
    nk1 = len(mapk1)
    nk2 = len(mapk2)
    cm1 = np.zeros((nk1, nk2, 4, 4, 4, 4), dtype=np.complex128)
    cm2 = np.zeros((nk1, nk2, 4, 4, 4, 4), dtype=np.complex128)
    for k in nb.prange(nk1):
        i = mapk1[k, 0]
        j = mapk1[k, 1]
        for kk in range(nk1):
            ii = mapk1[kk, 0]
            jj = mapk1[kk, 1]
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
                                cm1[k, kk, l1, l2, l3, l4] = cc / q1 * s11 * s21

    for k in nb.prange(nk2):
        i = mapk2[k, 0]
        j = mapk2[k, 1]
        for kk in range(nk2):
            ii = mapk2[kk, 0]
            jj = mapk2[kk, 1]
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
                                cm2[k, kk, l1, l2, l3, l4] = cc / q2 * s12 * s22

    return cm1, cm2


# @jit(nopython=True, parallel=True)
# def calc_overlap(n, nc, wfc, map1):
#     cm = np.zeros((nc, nc, 6, 6), dtype=np.complex128)
#     wfc_cut = np.zeros((nc, 6, 6), dtype=np.complex128)
#     for i in range(nc):
#         ix, iy = int(map1[i, 0]), int(map1[i, 1])
#         wfc_cut[i] = wfc[ix, iy, :, :]
#     wfc_cut_hc = np.zeros((nc, 6, 6), dtype=np.complex128)
#     for i in range(nc):
#         wfc_cut_hc[i] = np.conjugate(wfc_cut[i]).T.copy()
#     for i in nb.prange(nc):
#         mat_i = wfc_cut[i]
#         for j in range(i, nc):
#             mat_j_hc = wfc_cut_hc[j]
#             S = np.dot(mat_j_hc, mat_i)
#             for m in range(6):
#                 for n_band in range(6):
#                     val = S[m, n_band]
#                     cm[j, i, m, n_band] = val
#                     if i != j:
#                         cm[i, j, n_band, m] = np.conjugate(val)
#     return cm


def CoulMat(wfc, grid, dk2, a0):
    n = Config.N
    epsilon = Config.varepsilon
    print("epsilon: ", epsilon)
    epsilon0 = Config.varepsilon0
    print("epsilon0: ", epsilon0)
    e_charge = Config.e_charge
    # grid is an array contain kx and ky with N,N,2 dimensions
    cc = e_charge**2 / (2 * epsilon * epsilon0) * (dk2 / (4 * pi**2))  # ev/nm
    print("Coulomb constant: ", cc, "ev*nm")

    ############################ Use the circle cutoff grid with radius k cutoff
    # nkc, cutoffk, map0, map1 = cutoff_Grid_function(n, grid, a0)
    # print(nkc)
    # kArray = grid[cutoffk]
    # np.savetxt("./results/kgridcutoff.dat", kArray, delimiter=",", header="kx,ky")
    # with open("./results/circle_cutoff.dat", "w", newline="") as c:
    #     header = ["kx", "ky"]
    #     w = csv.DictWriter(c, fieldnames=header, delimiter=",")
    #     w.writeheader()
    #     for j in tqdm(range(n), desc="Cutoff circle"):
    #         for i in range(n):
    #             if cutoffk[i, j]:
    #                 rows = {}
    #                 rows["kx"] = grid[i, j, 0]
    #                 rows["ky"] = grid[i, j, 1]
    #                 w.writerow(rows)
    #     print("Calculate circle grid done")
    # overlap = calc_overlap(n, nkc, wfc, map1)
    ############################### The circle grid cutoff by Nguyen
    rcutoff = Config.rkcutoff
    mapk1, mapk2 = cutoff_grid_function_circle(n, grid, a0, rcutoff)
    cm1, cm2 = calcCMp_circle(mapk1, mapk2, cc, wfc, grid)
    print(f"Memory usage for CM1: {round(cm1.nbytes / 1024**2,2)} MB")
    print(f"Memory usage for CM2: {round(cm2.nbytes / 1024**2,2)} MB")
    ##################### Use the cutoff grid similar to the orginal BZ
    nc = int(2 * n / 18)
    shift = 0
    regionK1 = range(nc + 1 - shift, 2 * nc + 1 - shift)
    regionK2 = range(n - 2 * nc + 1 + shift, n - nc + 1 + shift)

    print("Number k cutoff:", nc)
    file1 = "./results/kgridcutoff1.dat"
    with open(file1, "w", newline="") as f1:
        print(f"Writing to {file1}...")
        header = ["kx", "ky"]
        w1 = csv.DictWriter(f1, fieldnames=header, delimiter=",")
        w1.writeheader()
        for j in tqdm(regionK1, desc="CM1"):
            for i in regionK1:
                ##### Write file for the K points cutoffed zone
                rows1 = {}
                rows1["kx"] = grid[i - 1, j - 1, 0]
                rows1["ky"] = grid[i - 1, j - 1, 1]
                w1.writerow(rows1)

    file2 = "./results/kgridcutoff2.dat"
    with open(file2, "w", newline="") as f2:
        print(f"Writing to {file2}...")
        header = ["kx", "ky"]
        w2 = csv.DictWriter(f2, fieldnames=header, delimiter=",")
        w2.writeheader()
        for j in tqdm(regionK2, desc="CM2"):
            for i in regionK2:
                ##### Write file for the K'' points cutoffed zone
                rows2 = {}
                rows2["kx"] = grid[i - 1, j - 1, 0]
                rows2["ky"] = grid[i - 1, j - 1, 1]
                w2.writerow(rows2)
    print("Writing completed successfully.")

    # cm1, cm2 = calcCMp(n, nc, cc, wfc, grid)
    # print(f"Memory usage for CM1: {round(cm1.nbytes / 1024**2,2)} MB")
    # print(f"Memory usage for CM2: {round(cm2.nbytes / 1024**2,2)} MB")

    # use_grid_circle = True
    # if use_grid_circle:
    #     return overlap, map1, nkc, grid
    # else:
    return cm1, cm2, mapk1, mapk2


def cutoff_grid_function_circle(N, grid, a0, rkcutoff) -> Tuple[NDArray, NDArray]:
    print("Radius cutoff k:", rkcutoff)
    list_k1 = []
    list_k2 = []

    kpoint = 4 * pi / (3 * a0)

    for j in tqdm(range(N), desc="Generating Cutoff Maps"):
        for i in range(N):
            kx = grid[i, j, 0]
            ky = grid[i, j, 1]

            dist_k1 = sqrt((kx + kpoint) ** 2 + ky**2)
            if dist_k1 <= rkcutoff:
                list_k1.append([i, j])

            dist_k2 = sqrt((kx - kpoint) ** 2 + ky**2)
            if dist_k2 <= rkcutoff:
                list_k2.append([i, j])

    map_k1 = np.array(list_k1, dtype=np.int64)
    map_k2 = np.array(list_k2, dtype=np.int64)

    print(f"Points in K1 region: {len(map_k1)}")
    print(f"Points in K2 region: {len(map_k2)}")
    print(f"Total kpoints in K1 and K2 region: {len(map_k1) + len(map_k2)}")

    file1 = "./results/circle1.dat"
    print(f"Writing to {file1}...")
    with open(file1, "w", newline="") as f1:
        header = ["kx", "ky"]
        w1 = csv.writer(f1)
        w1.writerow(header)
        for idx in tqdm(map_k1, desc="Writing CM1"):
            i, j = idx[0], idx[1]
            kx = grid[i, j, 0]
            ky = grid[i, j, 1]
            w1.writerow([kx, ky])

    file2 = "./results/circle2.dat"
    print(f"Writing to {file2}...")
    with open(file2, "w", newline="") as f2:
        header = ["kx", "ky"]
        w2 = csv.writer(f2)
        w2.writerow(header)
        for idx in tqdm(map_k2, desc="Writing CM2"):
            i, j = idx[0], idx[1]
            kx = grid[i, j, 0]
            ky = grid[i, j, 1]
            w2.writerow([kx, ky])

    print("Writing completed successfully.")

    return map_k1, map_k2


def cutoff_Grid_function(N, grid, a0) -> Tuple[int, NDArray[np.bool_]]:
    rkcutoff = Config.rkcutoff
    print("radius cutoff k:", rkcutoff)
    cutoffk = np.zeros([N, N], dtype=bool)
    nkcutoff = 0
    for j in tqdm(range(N), desc="Cutting off K-points"):
        for i in range(N):
            rk1 = sqrt((grid[i, j, 0] + 4 * pi / (3 * a0)) ** 2 + grid[i, j, 1] ** 2)
            rk2 = sqrt((grid[i, j, 0] - 4 * pi / (3 * a0)) ** 2 + grid[i, j, 1] ** 2)
            if rk1 <= rkcutoff:
                cutoffk[i, j] = True
            if rk2 <= rkcutoff:
                cutoffk[i, j] = True
            if cutoffk[i, j]:
                nkcutoff += 1
    count = 0
    map0 = np.zeros((N, N), dtype=np.int64)
    map1 = np.zeros((nkcutoff, 2), dtype=np.int64)
    for j in range(N):
        for i in range(N):
            if cutoffk[i, j]:
                map0[i, j] = count
                map1[count, 0] = i
                map1[count, 1] = j
                count += 1
    print("Number of k points after cutting off:", nkcutoff)
    return nkcutoff, cutoffk, map0, map1


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
                xi[:, :, m, n, 0] = const * (px[:, :, m, n] / (E[:, :, m] - E[:, :, n]))
                xi[:, :, m, n, 1] = const * (py[:, :, m, n] / (E[:, :, m] - E[:, :, n]))

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
