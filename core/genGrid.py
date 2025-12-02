from numpy import sqrt, pi
import numpy as np

from numpy.linalg import norm

from tqdm import tqdm


def Rhombus(alattice: float, N: int):
    G = 4 * pi / (sqrt(3) * alattice)
    ak1 = np.linspace(-G / 2, G / 2, N)
    ak2 = np.linspace(-G / 2, G / 2, N)

    akx = np.zeros([len(ak1), len(ak2)])
    aky = np.zeros([len(ak1), len(ak2)])
    kArr = np.zeros([len(ak1), len(ak2), 2])
    dkx = G / (N)
    dky = G / (N)
    for i in tqdm(range(N), desc="Create k grid", colour="red"):
        for j in range(N):
            akx[i][j] = sqrt(3) / 2 * (ak1[i] + ak2[j])
            aky[i][j] = -1 / 2 * (ak1[i] - ak2[j])

    kArr[:, :, 0] = akx
    kArr[:, :, 1] = aky

    return kArr, dkx, dky


def Monkhorst(alattice: float, N: int):
    G = 4 * pi / (sqrt(3) * alattice)
    b1 = np.array([sqrt(3) / 2 * G, -1 / 2 * G])
    b2 = np.array([sqrt(3) / 2 * G, 1 / 2 * G])
    akx = np.zeros((N, N))
    aky = np.zeros((N, N))

    kArr = np.zeros([N, N, 2])
    for i in tqdm(range(1, N + 1), desc="Monkhorst-Pack k grid", colour="red"):
        for j in range(1, N + 1):

            kvec = (2 * i - N - 1) / (2 * N) * b1 + (2 * j - N - 1) / (2 * N) * b2
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    # dkx = sqrt(3) / 2 * abs(akx[0, 0] - akx[1, 0])
    # dky = 1 / 2 * abs(akx[0, 1] - akx[0, 0])

    kArr[:, :, 0] = akx
    kArr[:, :, 1] = aky

    dkx = np.linalg.norm(b1) / N
    dky = np.linalg.norm(b2) / N

    return kArr, dkx, dky


def Cartesian(alattice: float, N: int):
    SR3 = sqrt(3)
    B = np.array(
        [
            [2.0 * pi / alattice, 2.0 * pi / alattice],
            [2.0 * pi / (SR3 * alattice), -2.0 * pi / (SR3 * alattice)],
        ]
    )

    ratio = np.linspace(0, 1, N)

    nk1_ratio = ratio.reshape(-1, 1)
    nk2_ratio = ratio.reshape(1, -1)

    grid_x = (B[0, 0] * nk1_ratio + B[0, 1] * nk2_ratio) - 2.0 * pi / alattice

    grid_y = (B[1, 0] * nk1_ratio + B[1, 1] * nk2_ratio) - 0.0

    kArr = np.stack((grid_x, grid_y), axis=-1)

    k1max = (2.0 * pi) / (SR3 * alattice)
    k2max = (2.0 * pi) / (SR3 * alattice)

    dk1 = 2.0 * k1max / (N - 1)
    dk2 = 2.0 * k2max / (N - 1)

    k1 = -k1max + np.arange(N) * dk1
    k2 = -k2max + np.arange(N) * dk2

    term1_x = SR3 * (k1[0] + k2[0]) / 2.0
    term2_x = SR3 * (k1[1] + k2[0]) / 2.0
    dkx = abs(term1_x - term2_x)

    term1_y = (k2[0] - k1[0]) / 2.0
    term2_y = (k2[0] - k1[1]) / 2.0
    dky = abs(term1_y - term2_y)

    return kArr, dkx, dky
