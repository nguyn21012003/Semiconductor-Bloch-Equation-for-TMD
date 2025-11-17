import csv
from typing import Tuple

from tqdm import tqdm

from core import genGrid
from core import pulse
from core.genHam import tbm_Hamiltonian
from numba import jit
import numba as nb
import numpy as np
from numpy import pi, sqrt
from numpy.typing import NDArray
import pandas as pd
from settings.configs import Config


def Hartree_Fock_func(data, modelNeighbor):
    alattice = data["alattice"]
    N = Config.N
    detuning = Config.detuning
    e_charge = Config.e_charge

    grid, dkx, dky = genGrid.Rhombus(alattice, N)
    print(f"dkx:{dkx},\ndky:{dky}")
    print(f"Grid dim:{grid.shape}")

    plot_grid(grid, "kgrid.dat")
    eigenValues, eigenVectors, pmx, pmy = solveHamiltonian(data, grid, modelNeighbor)

    egap = np.min(eigenValues[:, :, 1]) - np.max(eigenValues[:, :, 0])
    print(f"Energy gap: ", egap, "eV", "\n")
    print(f"Detuning: ", detuning, "eV")

    w0 = (egap) / Config.hbar
    print(f"Frequency: ", w0)
    # vector_pot = pulse.At(w0)

    check_degenaracy = np.zeros([3, 3])
    for i in tqdm(range(N), desc="Check degenaracy"):
        for j in range(N):
            for nu in range(0, 3):
                for mu in range(0, 3):
                    if np.abs(eigenValues[i, j, nu] - eigenValues[i, j, mu]) < 1e-2:
                        check_degenaracy[mu, nu] += 1
    dk = (dkx, dky)
    dk2 = dkx * dky
    # print(check_degenaracy)
    ################################################################# Post SBE
    # p = momentum(eigenVectors, grid, data, modelNeighbor)
    p = (pmx, pmy)
    xi = dipole(eigenValues, p, check_degenaracy)
    xiovx = np.zeros([N, N, 3, 3], dtype="complex")
    xiovy = np.zeros([N, N, 3, 3], dtype="complex")
    xiovx[:, :, :, :] = (xi[:, :, :, :, 0] + xi[:, :, :, :, 1]) / sqrt(3)
    xiovy[:, :, :, :] = (-xi[:, :, :, :, 0] + xi[:, :, :, :, 1]) / sqrt(3)

    ################################################################# SBE
    rho = np.zeros([N, N, 3, 3], dtype="complex")
    # note: chua co spin nen chi xet la 2, co spin thi la 4
    rho[:, :, 0, 0] = 1.0 + 0.0j
    # rho[:, :, 1, 1] = 0.0 + 0.0j

    tmin = Config.tmin
    print(f"tmin:", tmin)
    tmax = Config.tmax
    print(f"tmax:", tmax)
    dt = Config.dt
    tL = Config.time_duration
    E0 = Config.E0
    ntmax = int((tmax - tmin) / dt)

    # data_Coulomb = Coulomb(alattice, eigenVectors, grid, dk)
    data_Coulomb = None
    ############################################################ Calculating SBE
    P_t_array = np.zeros([ntmax, 2], dtype="complex")  # Save P(t)
    time_array = np.zeros(ntmax, dtype=float)
    Atx, Aty = 0, 0
    t = tmin
    for nt in tqdm(range(ntmax), desc="RK4 SBE: "):
        Etx, Ety = pulse.Et(t, E0, tL)
        Etime = (Etx, Ety)
        Atx = Config.e_charge / Config.m_e * (Atx - Etx * dt)
        Aty = Config.e_charge / Config.m_e * (Aty - Ety * dt)
        Atime = (Atx, Aty)
        rho = rk4(rho, t, dt, Etime, Atime, xi, p, dk, data_Coulomb, eigenValues, w0)
        sum1 = 0j
        sum2 = 0j
        for jb in [0]:
            for ib in [1, 2]:
                sum1 += np.sum(xiovx[:, :, jb, ib] * rho[:, :, ib, jb])
                sum2 += np.sum(xiovy[:, :, jb, ib] * rho[:, :, ib, jb])

        Px = -sum1 * dk2 / (4.0 * pi**2)
        Py = -sum2 * dk2 / (4.0 * pi**2)

        time_array[nt] = t
        P_t_array[nt, 0] = Px
        P_t_array[nt, 1] = Py
        t += dt

    print(time_array)
    absorptionSpectra(P_t_array, egap, time_array)


def absorptionSpectra(P_t_array, egap, evoltime):
    hbar = Config.hbar
    dt = Config.dt
    E0 = Config.E0
    tL = Config.time_duration
    omegaL = egap / hbar
    omega = (egap - 0.5) / hbar
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


def totalPolarizationFunction(rho, xi, dk):
    qe = Config.e_charge
    xi_x, xi_y = xi
    dkx, dky = dk
    c = dkx * dky / (2 * pi) ** 2
    N = Config.N
    s1 = 0.0
    s2 = 0.0
    for nk1 in range(N):
        for nk2 in range(N):
            for nu in range(3):
                for mu in range(nu + 1, 3):
                    s1 += 2.0 * np.real(xi_x[nk1, nk2, mu, nu] * rho[nk1, nk2, nu, mu])
                    s2 += 2.0 * np.real(xi_y[nk1, nk2, mu, nu] * rho[nk1, nk2, nu, mu])

    Px_t = qe * s1 * c
    Py_t = qe * s2 * c
    return Px_t, Py_t


def rk4(rho, t, dt, Et, At, xi, p, dk, exchange_term, E, w0):
    k1 = rhsSBE(rho, t, Et, At, xi, p, dk, E, w0, exchange_term)
    k2 = rhsSBE(rho + 0.5 * k1 * dt, t, Et, At, xi, p, dk, E, w0, exchange_term)
    k3 = rhsSBE(rho + 0.5 * k2 * dt, t, Et, At, xi, p, dk, E, w0, exchange_term)
    k4 = rhsSBE(rho + 1.0 * k3 * dt, t, Et, At, xi, p, dk, E, w0, exchange_term)

    return rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rhsSBE(rho, t, Et, At, xiov, p, dk, E, omega_L, exchange_term):
    Etx, Ety = Et
    N = Config.N
    Y = np.zeros((N, N, 3, 3), dtype="complex")
    T2 = Config.T_cohenrent
    eE1 = Etx / (sqrt(3))
    eE2 = Etx / (sqrt(3))
    # xi_1 = sqrt(3) / 2 * xi_x - 0.5 * xi_y
    # xi_2 = sqrt(3) / 2 * xi_x + 0.5 * xi_y
    # hfse = hamiltonian_coulomb(exchange_term, rho, N)
    for m in range(1):
        for n in range(1, 3):
            omegaCV = (E[:, :, n] - E[:, :, m]) / Config.hbar
            omegaRabi = (Etx * xiov[:, :, n, m, 0] + Ety * xiov[:, :, n, m, 1]) / Config.hbar
            # omegaRabi = (Atx * px[:, :, n, m] + Aty * py[:, :, n, m]) / (Config.hbar * me)
            # omegaHF = hfse[:, :, n, m] / Config.hbar
            Y[:, :, n, m] = -1j * (omegaCV - omega_L) * rho[:, :, n, m] - 1j * omegaRabi - rho[:, :, n, m] / T2
            Y[:, :, m, n] = np.conjugate(Y[:, :, n, m])

    return Y


def plot_grid(kArr, file):
    kx = kArr[:, :, 0].flatten()
    ky = kArr[:, :, 1].flatten()
    np.savetxt(f"./results/" + file, np.column_stack((kx, ky)), header="kx,ky", delimiter=",")


@jit(nopython=True, parallel=True)
def hamiltonian_coulomb(data, rho, N):
    (
        coulomb_const,
        num_kcutoff,
        mapping,
        grid,
        V_coulomb,
        check_valid_cutoff_array,
    ) = data
    hamiltonian_coulomb = np.zeros((N, N, 3, 3), dtype=np.complex128)
    for i in nb.prange(num_kcutoff):
        for mu in range(0, 2):
            for nu in range(mu, 2):
                sum1 = 0.0j
                for j in range(num_kcutoff):
                    state1 = check_valid_cutoff_array[mapping[i, 0], mapping[i, 1], 0]
                    state2 = check_valid_cutoff_array[mapping[j, 0], mapping[j, 1], 0]

                    state3 = check_valid_cutoff_array[mapping[i, 0], mapping[i, 1], 1]
                    state4 = check_valid_cutoff_array[mapping[j, 0], mapping[j, 1], 1]

                    if not (state1 and state2 or state3 and state4):
                        continue
                    if i == j:
                        continue
                    kxi = grid[mapping[i, 0], mapping[i, 1], 0]
                    kxj = grid[mapping[j, 0], mapping[j, 1], 0]
                    kyi = grid[mapping[i, 0], mapping[i, 1], 1]
                    kyj = grid[mapping[j, 0], mapping[j, 1], 1]
                    q = 1.0 / (np.sqrt((kxj - kxi) ** 2 + (kyj - kyi) ** 2))

                    for beta in range(2):
                        for alpha in range(2):
                            if alpha != beta:
                                sum1 -= (V_coulomb[i, j, mu, beta] * rho[mapping[j, 0], mapping[j, 1], beta, alpha] * V_coulomb[j, i, alpha, nu]) * q
                hamiltonian_coulomb[mapping[i, 0], mapping[i, 1], mu, nu] = sum1 * coulomb_const
                if nu != mu:
                    hamiltonian_coulomb[mapping[i, 0], mapping[i, 1], nu, mu] = np.conjugate(hamiltonian_coulomb[mapping[i, 0], mapping[i, 1], mu, nu])

    return hamiltonian_coulomb


def Coulomb(alattice: str, eigenVectors: NDArray[np.complex128], grid, dk):
    dkx, dky = dk
    N = Config.N
    epsilon = Config.varepsilon
    epsilon0 = Config.varepsilon0
    e_charge = Config.e_charge

    # grid is an array contain kx and ky with N,N,2 dimensions
    # dk (1/nm)

    ratio = dkx * dky / (2 * pi) ** 2
    coulomb_const = e_charge**2 / (2 * epsilon * epsilon0) * ratio  # ev/nm
    print(f"Coulomb constant: ", coulomb_const, "ev/nm")
    print(f"Ratio:", ratio)

    print(f"Grid before cutting off:", grid.shape)
    num_kcutoff, check_valid_cutoff_array = cutoff_Grid_function(N, grid, alattice)
    try:
        indices_k1_tuple = np.where(check_valid_cutoff_array[:, :, 0])
        # check True elements and return index

        map_k1 = np.column_stack(indices_k1_tuple)
        num_k1_points = len(map_k1)
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

    kArray = grid[mappingArray[:, 0], mappingArray[:, 1], :]
    np.savetxt("./results/kgridcutoff.dat", kArray, delimiter=",", header="kx,ky")
    coulombInteractionArray = np.zeros([num_kcutoff, num_kcutoff, 3, 3], dtype=complex)
    # new Coulomb with k cutoff

    vecs_at_cutoff = eigenVectors[mappingArray[:, 0], mappingArray[:, 1], :, :3]
    # for i in tqdm(range(num_kcutoff), desc="Calculating sum eigenvectors"):
    #     vec_i = vecs_at_cutoff[i]  # (4, num_orbitals)
    #     is_k_i_in_K1 = i < len(map_k1)
    #     for j in range(num_kcutoff):
    #         is_k_j_in_K1 = j < len(map_k1)
    #         if is_k_i_in_K1 == is_k_j_in_K1:
    #             vec_j = vecs_at_cutoff[j]  # (4, num_orbitals)
    #             overlap = np.conjugate(vec_i).T @ vec_j
    #             coulombInteractionArray[i, j, :, :] = overlap

    for i in tqdm(range(num_kcutoff), desc="Calculating sum eigenvectors"):
        for j in range(num_kcutoff):
            state1 = check_valid_cutoff_array[mappingArray[i, 0], mappingArray[i, 1], 0]
            state2 = check_valid_cutoff_array[mappingArray[j, 0], mappingArray[j, 1], 0]

            state3 = check_valid_cutoff_array[mappingArray[i, 0], mappingArray[i, 1], 1]
            state4 = check_valid_cutoff_array[mappingArray[j, 0], mappingArray[j, 1], 1]
            if state1 and state2 or state3 and state4:
                cj = eigenVectors[mappingArray[j, 0], mappingArray[j, 1], :, :]
                ci = eigenVectors[mappingArray[i, 0], mappingArray[i, 1], :, :]
                S = np.conjugate(cj).T @ ci
                for m in range(2):
                    for n in range(2):
                        coulombInteractionArray[j, i, m, n] = S[m, n]

    # print(coulombInteractionArray)

    exch = (
        coulomb_const,
        num_kcutoff,
        mappingArray,
        # kArray,
        grid,
        coulombInteractionArray,
        check_valid_cutoff_array,
    )
    return exch


def dipole(E, p, check_degenaracy):
    N = Config.N
    hbar = Config.hbar
    me = Config.m_e
    px, py = p
    xi = np.zeros([N, N, 3, 3, 2], dtype="complex")

    const = -1j * hbar / me
    for m in tqdm([0, 1, 2], desc="Dipole"):
        for n in [0, 1, 2]:
            if m != n:
                xi[:, :, m, n, 0] = const * (px[:, :, m, n] / (E[:, :, m] - E[:, :, n]))
                xi[:, :, m, n, 1] = const * (py[:, :, m, n] / (E[:, :, m] - E[:, :, n]))

                xi[:, :, n, m, 0] = np.conjugate(xi[:, :, m, n, 0])
                xi[:, :, n, m, 1] = np.conjugate(xi[:, :, m, n, 1])

    # return xi_x, xi_y
    return xi


def momentum_old(eigenVectors, grid, data, modelNeighbor):
    N = Config.N
    me = Config.m_e
    hbar = Config.hbar
    alattice = data["alattice"]

    px = np.zeros([N, N, 3, 3], dtype="complex")
    py = np.zeros([N, N, 3, 3], dtype="complex")
    # for i in tqdm(range(N), desc="Calculating momentum"):
    #     for j in range(N):
    for i in tqdm(range(N), desc="Calculating momentum"):
        for j in range(N):
            alpha = grid[i, j, 0] / 2 * alattice
            beta = sqrt(3) / 2 * grid[i, j, 1] * alattice

            _, dhkx, dhky = tbm_Hamiltonian(alpha, beta, data, modelNeighbor, alattice)
            c = eigenVectors[i, j, :, :]
            cdagger = np.conjugate(c)
            px[i, j, :, :] = cdagger @ (dhkx[:, :] @ c)
            py[i, j, :, :] = cdagger @ (dhky[:, :] @ c)
    px = px * (me / hbar)
    py = py * (me / hbar)

    return px, py


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
            rk1 = sqrt((grid[i, j, 0] - 4 * pi / (3 * alattice)) ** 2 + grid[i, j, 1] ** 2)
            rk2 = sqrt((grid[i, j, 0] + 4 * pi / (3 * alattice)) ** 2 + grid[i, j, 1] ** 2)
            if rk1 < rkcutoff:
                check_valid_cutoff_grid[i, j, 0] = True
                count += 1
            elif rk2 < rkcutoff:
                check_valid_cutoff_grid[i, j, 1] = True
                count += 1
    return count, check_valid_cutoff_grid


def solveHamiltonian_Numerical(data, kArray, modelNeighbor, delta_k: float = 1e-6) -> Tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    N = Config.N
    alattice = data["alattice"]

    eigenValues = np.zeros([N, N, 3])
    eigenVectors = np.zeros([N, N, 3, 3], dtype="complex")
    dH_kx = np.zeros([N, N, 3, 3], dtype="complex")
    dH_ky = np.zeros([N, N, 3, 3], dtype="complex")

    a_over_2 = alattice / 2.0
    sqrt3_a_over_2 = sqrt(3) / 2.0 * alattice

    for i in tqdm(range(N), desc="Calculate bandstructure (Numerical Grad)"):
        for j in range(N):
            kx = kArray[i, j, 0]
            ky = kArray[i, j, 1]

            alpha_mid = kx * a_over_2
            beta_mid = ky * sqrt3_a_over_2

            ham, _, _ = tbm_Hamiltonian(alpha_mid, beta_mid, data, modelNeighbor, alattice)

            vals, vecs = np.linalg.eigh(ham)
            eigenVectors[i, j, :, :] = vecs
            eigenValues[i, j, :] = vals

            alpha_plus_x = (kx + delta_k) * a_over_2
            ham_plus_x, _, _ = tbm_Hamiltonian(alpha_plus_x, beta_mid, data, modelNeighbor, alattice)

            alpha_minus_x = (kx - delta_k) * a_over_2
            ham_minus_x, _, _ = tbm_Hamiltonian(alpha_minus_x, beta_mid, data, modelNeighbor, alattice)

            dH_kx[i, j, :, :] = (ham_plus_x - ham_minus_x) / (2.0 * delta_k)

            beta_plus_y = (ky + delta_k) * sqrt3_a_over_2
            ham_plus_y, _, _ = tbm_Hamiltonian(alpha_mid, beta_plus_y, data, modelNeighbor, alattice)

            beta_minus_y = (ky - delta_k) * sqrt3_a_over_2
            ham_minus_y, _, _ = tbm_Hamiltonian(alpha_mid, beta_minus_y, data, modelNeighbor, alattice)

            dH_ky[i, j, :, :] = (ham_plus_y - ham_minus_y) / (2.0 * delta_k)

            del vals, vecs, ham
            del ham_plus_x, ham_minus_x, ham_plus_y, ham_minus_y

    return eigenValues, eigenVectors, dH_kx, dH_ky


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

    eigenValues = np.zeros([N, N, 3])
    eigenVectors = np.zeros([N, N, 3, 3], dtype="complex")
    pmx = np.zeros([N, N, 3, 3], dtype="complex")
    pmy = np.zeros([N, N, 3, 3], dtype="complex")
    # for i in tqdm(range(N), desc="Calculate bandstructure"):
    #     for j in range(N):
    #         alpha = kArray[i, j, 0] / 2 * alattice
    #         beta = sqrt(3) / 2 * kArray[i, j, 1] * alattice
    #         ham, dhkx, dhky = tbm_Hamiltonian(alpha, beta, data, modelNeighbor, alattice)
    #         vals, vecs = np.linalg.eigh(ham)
    #         eigenVectors[i, j, :, :] = vecs
    #         eigenValues[i, j, :] = vals
    #         dH_kx[i, j, :, :] = dhkx
    #         dH_ky[i, j, :, :] = dhky
    #         del vals, vecs, dhkx, dhky

    for j in tqdm(range(N), desc="Calculate bandstructure"):
        for i in range(N):
            alpha = kArray[i, j, 0] / 2 * alattice
            beta = sqrt(3) / 2 * kArray[i, j, 1] * alattice
            ham, dhkx, dhky = tbm_Hamiltonian(alpha, beta, data, modelNeighbor, alattice)
            vals, vecs = np.linalg.eigh(ham)
            for jb in range(0, 3):
                eigenValues[i, j, jb] = vals[jb]
                for ib in range(0, 3):
                    sum1 = 0j
                    sum2 = 0j
                    for jjb in range(0, 3):
                        for iib in range(0, 3):
                            sum1 += np.conjugate(vecs[iib, ib]) * dhkx[iib, jjb] * vecs[jjb, jb]
                            sum2 += np.conjugate(vecs[iib, ib]) * dhky[iib, jjb] * vecs[jjb, jb]
                    pmx[i, j, ib, jb] = sum1 * me / hbar
                    pmy[i, j, ib, jb] = sum2 * me / hbar

    return eigenValues, eigenVectors, pmx, pmy
