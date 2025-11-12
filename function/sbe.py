from typing import Tuple

import numpy as np
from numpy import pi, sqrt
from numpy.typing import NDArray
from tqdm import tqdm

from core import genGrid
from core import pulse
from core.genHam import tbm_Hamiltonian
from settings.configs import Config
from numba import jit
import numba as nb


def Hartree_Fock_func(data, modelNeighbor):
    alattice = data["alattice"] * 1e-1
    N = Config.N
    detuning = Config.detuning
    e_charge = Config.e_charge

    grid, dkx, dky = genGrid.Monkhorst(alattice, N)
    print(grid.shape)
    plot_grid(grid, "kgrid.dat")
    eigenValues, eigenVectors, grad_Ham_kx, grad_Ham_ky = solveHamiltonian(data, grid, modelNeighbor)

    egap = np.min(eigenValues[:, :, 1]) - np.max(eigenValues[:, :, 0])
    print(f"Energy gap: ", egap, "eV", "\n")
    print(f"Detuning: ", detuning, "eV")

    w0 = (egap + detuning) * e_charge / Config.hbar
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
    # print(check_degenaracy)
    ################################################################# Post SBE
    p = momentum(eigenVectors, grad_Ham_kx, grad_Ham_ky)
    xi = dipole(eigenValues, p, check_degenaracy)

    ################################################################# SBE
    rho = np.zeros([N, N, 3, 3])  # note: chua co spin nen chi xet la 2, co spin thi la 4
    rho[:, :, 0, 0] = 1
    # rho[:, :, 1, 1] = 1
    tmin = Config.tmin
    tmax = Config.tmax
    dt = Config.dt
    ntmax = int((tmax - tmin) / dt)
    T02 = Config.T_cohenrent

    data_Coulomb = Coulomb(alattice, eigenVectors, grid, dk)
    # ############################################################ Calculating SBE
    # P_t_array = np.zeros([ntmax, 2], dtype=float)  # Save P(t)
    # E_t_array = np.zeros([ntmax, 2], dtype=float)  # Save E(t)
    # time_array = np.zeros(ntmax, dtype=float)
    # Atx, Aty = 0, 0
    # At = (Atx, Aty)
    # for nt in tqdm(range(0, ntmax), desc="RK4 SBE: "):
    #     time = tmin + ((nt + 1) * dt)
    #     Etx, Ety = pulse.Et(time, w0)
    #     Et = (Etx, Ety)
    #     rk4(rho, At, Et, p, dk, data_Coulomb, eigenValues, T02)
    #     Px, Py = totalPolarizationFunction(rho, xi, dk)
    #     P_t_array[nt, 0] = Px
    #     P_t_array[nt, 1] = Py
    # absorptionSpectra(P_t_array, w0, egap)


def absorptionSpectra(P_t_array: NDArray, w0, egap):  # Mảng P(t) (kích thước [ntmax, 2])
    Nw = 2000
    tmin = Config.tmin
    tmax = Config.tmax
    dt = Config.time_duration
    ntmax = int((tmax - tmin) / dt)
    E0 = Config.E0
    epl = Config.epl
    hbar = Config.hbar
    dt = Config.dt

    wa = w0 - 1.0 / hbar  # (đơn vị 1/fs)
    wb = w0 + 1.0 / hbar  # (đơn vị 1/fs)
    dw = (wb - wa) / Nw
    omega_array = wa + np.arange(Nw + 1) * dw
    output_file = "AbSpect.dat"
    results = []

    for w in tqdm(omega_array, desc="Calculating Spectrum (Manual FFT)"):
        s1, s2, s3, s4, s5, s6 = (0j, 0j, 0j, 0j, 0j, 0j)
        for tt in range(ntmax + 1):
            t = tmin + (tt) * dt
            (Ex, Ey) = pulse.Et(t, w0)
            if np.abs(Ex) < 1e-16:
                theta = pi / 2.0
            else:
                theta = np.arctan(Ey / Ex)
            eiwt = np.exp(1j * w * t)
            try:
                s1 += eiwt * P_t_array[tt, 0] * dt
                s2 += eiwt * P_t_array[tt, 1] * dt

                s3 += eiwt * Ex * dt
                s4 += eiwt * Ey * dt

                E_norm = np.sqrt(Ex**2 + Ey**2)
                P_dot_E = (P_t_array[tt, 0] * Ex) + (P_t_array[tt, 1] * Ey)
                s5 += eiwt * (P_dot_E / E_norm) * dt

                E_mag_proj = (Ex * np.cos(theta)) + (Ey * np.sin(theta))
                s6 += eiwt * E_mag_proj * dt

            except IndexError:
                if tt == ntmax:
                    print(f"Cảnh báo: P_t_array ngắn hơn (ntmax) so với vòng lặp (ntmax+1).")
                break

        chi_x = s1 / (s3 + 1e-16)
        chi_y = s2 / (s4 + 1e-16)
        chi_s56 = s5 / (s6 + 1e-16)
        results.append(
            (
                w * hbar,
                w * hbar - egap,
                np.real(s1),
                np.imag(s1),
                np.real(s2),
                np.imag(s2),
                np.real(s3),
                np.imag(s3),
                np.real(s4),
                np.imag(s4),
                np.real(chi_x),
                np.imag(chi_x),
                np.real(chi_y),
                np.imag(chi_y),
                np.real(s5),
                np.imag(s5),
                np.real(s6),
                np.imag(s6),
                np.real(chi_s56),
                np.imag(chi_s56),
            )
        )

    # 8. Ghi file
    header = (
        "Energy(eV),Energy_minus_Egap(eV),"
        "Re(Px_w),Im(Px_w),Re(Py_w),Im(Py_w),"
        "Re(Ex_w),Im(Ex_w),Re(Ey_w),Im(Ey_w),"
        "Re(Chi_x),Im(Chi_x),Re(Chi_y),Im(Chi_y),"
        "Re(S5_P_dot_e),Im(S5_P_dot_e),"
        "Re(S6_E_mag),Im(S6_E_mag),"
        "Re(S5/S6),Im(S5/S6)"
    )

    np.savetxt(output_file, np.array(results), fmt="%.15e", delimiter=",", header=header)

    return results


def totalPolarizationFunction(rho, xi, dk):
    qe = Config.e_charge

    xi_x, xi_y = xi
    dkx, dky = dk
    c = dkx * dky / (2 * pi) ** 2

    Px_t = qe * c * np.real(np.einsum("ijmn,ijnm->", xi_x, rho))
    Py_t = qe * c * np.real(np.einsum("ijmn,ijnm->", xi_y, rho))
    return Px_t, Py_t


def rk4(rho, At, Et, p, dk, exchange_term, E, T02):
    dt = Config.dt
    k1 = rhsSBE(0, rho, At, Et, p, dk, E, T02, exchange_term)
    k2 = rhsSBE(0.5, rho + 0.5 * k1 * dt, At, Et, p, dk, E, T02, exchange_term)
    k3 = rhsSBE(0.5, rho + 0.5 * k2 * dt, At, Et, p, dk, E, T02, exchange_term)
    k4 = rhsSBE(1.0, rho + 1.0 * k3 * dt, At, Et, p, dk, E, T02, exchange_term)

    return rho + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rhsSBE(ind, rho, At, Et, p, dk, E, T02, exchange_term):
    Atx, Aty = At
    Etx, Ety = Et
    px, py = p
    dkx, dky = dk
    qe = Config.e_charge
    me = Config.m_e
    dt = Config.dt
    N = Config.N
    H = np.zeros([N, N, 3, 3], dtype="complex")
    den_of_electron = (dkx * dky / (2 * pi) ** 2) * np.real(np.sum(rho[:, :, 1, 1]))
    T2 = (1 / T02) + den_of_electron * Config.gamma

    for m in range(3):
        H[:, :, m, m] = E[:, :, m]

    Ax = Atx - ind * Etx * dt
    Ay = Aty - ind * Ety * dt
    HF = hamiltonian_coulomb(exchange_term, rho, N) - (qe / me * Ax * px) - (qe / me * Ay * py)
    H += HF
    Y = -(1j / Config.hbar) * (H @ rho - rho @ H) - rho * (1 - np.eye(3)) * T2
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
        num_k1_points,
        mapping,
        kArray,
        V_coulomb,
    ) = data
    hamiltonian_coulomb = np.zeros((N, N, 3, 3), dtype=np.complex128)
    for i in nb.prange(num_kcutoff):
        is_k_i_in_K1 = i < num_k1_points
        for mu in range(0, 3):
            for nu in range(0, 3):
                sum1 = 0.0j
                for j in range(num_kcutoff):
                    is_k_j_in_K1 = j < num_k1_points
                    if is_k_i_in_K1 != is_k_j_in_K1:
                        continue
                    if i == j:
                        continue
                    c = kArray[i] - kArray[j]
                    q = 1 / (np.linalg.norm(c) + 1e-12)
                    V_coulomb_ji = V_coulomb[j, i, :, nu]
                    V_coulomb_ij = V_coulomb[i, j, mu, :]

                    term = 0.0j
                    for beta in range(3):
                        for alpha in range(3):
                            term += V_coulomb[i, j, mu, beta] * rho[mapping[j, 0], mapping[j, 1], beta, alpha] * V_coulomb[j, i, alpha, nu]
                    sum1 -= term * q
                hamiltonian_coulomb[mapping[i, 0], mapping[i, 1], mu, nu] = sum1 * coulomb_const

    return hamiltonian_coulomb


def Coulomb(alattice: str, eigenVectors: NDArray[np.complex128], grid, dk):
    dkx, dky = dk
    N = Config.N
    epsilon = Config.varepsilon
    epsilon0 = Config.varepsilon0
    e_charge = Config.e_charge

    # grid is an array contain kx and ky with N,N,2 dimensions
    # dk (1/nm)

    coulomb_const = e_charge**2 / (2 * epsilon * epsilon0) * (dkx * dky / (2 * pi) ** 2)  # J/m
    print(f"Coulomb constant: ", coulomb_const, "J/m")

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
    print(kArray.shape)
    np.savetxt("./results/kgridcutoff.dat", kArray, delimiter=",", header="kx,ky")
    coulombInteractionArray = np.zeros([num_kcutoff, num_kcutoff, 3, 3], dtype=complex)
    # new Coulomb with k cutoff

    vecs_at_cutoff = eigenVectors[mappingArray[:, 0], mappingArray[:, 1], :, :3]
    for i in tqdm(range(num_kcutoff), desc="Calculating sum eigenvectors"):
        vec_i = vecs_at_cutoff[i]  # (4, num_orbitals)
        is_k_i_in_K1 = i < len(map_k1)
        for j in range(num_kcutoff):
            is_k_j_in_K1 = j < len(map_k1)
            if is_k_i_in_K1 == is_k_j_in_K1:
                vec_j = vecs_at_cutoff[j]  # (4, num_orbitals)
                overlap = np.conjugate(vec_i).T @ vec_j
                coulombInteractionArray[i, j, :, :] = overlap

    # print(coulombInteractionArray)

    exch = (
        coulomb_const,
        num_kcutoff,
        num_k1_points,
        mappingArray,
        kArray,
        coulombInteractionArray,
    )
    return exch


def dipole(E, p, check_degenaracy):
    N = Config.N
    hbar = Config.hbar
    me = Config.m_e
    px, py = p
    xi_x = np.zeros([N, N, 3, 3], dtype="complex")
    xi_y = np.zeros([N, N, 3, 3], dtype="complex")
    xi = np.zeros([N, N, 3, 3, 2], dtype="complex")

    const = -1j * hbar / me
    for i in tqdm(range(N), desc="Calculating dipole"):
        for j in range(N):
            for m in range(0, 3):
                for n in range(m + 1, 3):
                    if check_degenaracy[m, n] == 0:
                        xi_x[i, j, m, n] = const * (px[i, j, m, n] / (E[i, j, m] - E[i, j, n]))
                        xi_y[i, j, m, n] = const * (py[i, j, m, n] / (E[i, j, m] - E[i, j, n]))

                        xi_x[i, j, n, m] = np.conjugate(xi_x[i, j, m, n])
                        xi_y[i, j, n, m] = np.conjugate(xi_y[i, j, m, n])

    xi[:, :, :, :, 0] = xi_x
    xi[:, :, :, :, 1] = xi_y
    return xi_x, xi_y


def momentum(eigenVectors, grad_Ham_kx, grad_Ham_ky):

    N = Config.N
    me = Config.m_e
    hbar = Config.hbar

    px = np.zeros([N, N, 3, 3], dtype="complex")
    py = np.zeros([N, N, 3, 3], dtype="complex")
    for i in tqdm(range(N), desc="Calculating momentum"):
        for j in range(N):
            c = eigenVectors[i, j, :, :]
            cdagger = np.conjugate(c).T
            px[i, j, :, :] = cdagger @ (grad_Ham_kx[i, j, :, :] @ c)
            py[i, j, :, :] = cdagger @ (grad_Ham_ky[i, j, :, :] @ c)
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


def solveHamiltonian(data, kArray, modelNeighbor) -> Tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Solve the tight-binding Hamiltonian and its momentum derivatives for each k-point.

    This function constructs and diagonalizes the tight-binding Hamiltonian `H(k)`
    at each k-point in the 2D grid. In addition to the eigenvalues
    and eigenvectors, it also returns the derivatives of the Hamiltonian with respect
    to k_x and k_y (∂H/∂k_x, ∂H/∂k_y), which are often used to compute
    velocity operators, Berry curvature, or optical matrix elements.

    Parameters
    ----------
    data : dict
        A dictionary containing model parameters, typically including:
        - "alattice" : float
            The lattice constant a of the system.
        Other parameters depend on the implementation of `tbm_Hamiltonian`.

    kArray : ndarray of shape (N, N, 2)
        The 2D Monkhorst–Pack grid of k-points.
        Each element `kArray[i, j] = [k_x, k_y]` represents one k-point
        in the reciprocal space.

    modelNeighbor : Any
        Data structure containing hopping parameters or neighbor information,
        passed to `tbm_Hamiltonian` for constructing the Hamiltonian matrix.

    Returns
    -------
    eigenValues : ndarray of shape (N, N, 3)
        The energy eigenvalues (band energies) for each k-point.
        The third dimension corresponds to the number of bands.

    eigenVectors : ndarray of shape (N, N, 3, 3), complex
        The normalized eigenvectors of the Hamiltonian at each k-point.
        The third index corresponds to orbital components,
        and the fourth to band indices.

    dhkx : ndarray of shape (3, 3), complex
        The derivative of the Hamiltonian with respect to k_x (∂H/∂k_x)
        at the last computed k-point. This is useful for calculating
        the velocity operator v_x = (1/ħ) ∂H/∂k_x.

    dhky : ndarray of shape (3, 3), complex
        The derivative of the Hamiltonian with respect to k_y (∂H/∂k_y)
        at the last computed k-point. This is useful for calculating
        the velocity operator v_y = (1/ħ) ∂H/∂k_y.

    Notes
    -----
    - The Hamiltonian and its derivatives are obtained from:
        `ham, dhkx, dhky = tbm_Hamiltonian(alpha, beta, data, modelNeighbor, alattice)`
    - Phase factors are defined as:
        α = (k_x * a) / 2, β = (√3 / 2) * (k_y * a)
    - The diagonalization uses `numpy.linalg.eigh`, assuming the Hamiltonian is Hermitian.
    - Currently, `dhkx` and `dhky` returned correspond to the *last evaluated* k-point
      in the loop. To obtain them for all k-points, consider storing them in arrays.

    Example
    -------
    >>> eigenVals, eigenVecs, dhkx, dhky = solveHamiltonian(data, kArray, modelNeighbor)
    >>> print(eigenVals.shape)
    (N, N, 3)
    >>> print(eigenVecs.shape)
    (N, N, 3, 3)
    >>> print(dhkx.shape, dhky.shape)
    (3, 3) (3, 3)
    """
    N = Config.N
    alattice = data["alattice"]
    # grid K phai chay lai moi lan do khac alattice

    eigenValues = np.zeros([N, N, 3])
    eigenVectors = np.zeros([N, N, 3, 3], dtype="complex")
    dH_kx = np.zeros([N, N, 3, 3], dtype="complex")
    dH_ky = np.zeros([N, N, 3, 3], dtype="complex")
    for i in tqdm(range(N), desc="Calculate bandstructure"):
        for j in range(N):
            alpha = kArray[i, j, 0] / 2 * alattice
            beta = sqrt(3) / 2 * kArray[i, j, 1] * alattice
            ham, dhkx, dhky = tbm_Hamiltonian(alpha, beta, data, modelNeighbor, alattice)
            vals, vecs = np.linalg.eigh(ham)
            eigenVectors[i, j, :, :] = vecs  # C[i, j, orb, λ]
            eigenValues[i, j, :] = vals
            dH_kx[i, j, :, :] = dhkx
            dH_ky[i, j, :, :] = dhky
            del vals, vecs, dhkx, dhky

    return eigenValues, eigenVectors, dH_kx, dH_ky
