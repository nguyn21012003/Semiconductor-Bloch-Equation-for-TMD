from numpy import sqrt, pi
import numpy as np

from tqdm import tqdm


def Rhombus(alattice: float, N: int):
    G = 4 * pi / (sqrt(3) * alattice)
    ak1 = np.linspace(-G / 2, G / 2, N)
    ak2 = np.linspace(-G / 2, G / 2, N)

    akx = np.zeros([len(ak1), len(ak2)])
    aky = np.zeros([len(ak1), len(ak2)])
    dkx = G / (N - 1)
    dky = G / (N - 1)
    for i in tqdm(range(N), desc="Create k grid", colour="red"):
        for j in range(N):
            akx[i][j] = sqrt(3) / 2 * (ak1[i] + ak2[j])
            aky[i][j] = -1 / 2 * (ak1[i] - ak2[j]) * 0

    return akx, aky, dkx, dky


def Monkhorst(alattice: float, N: int):
    G = 4 * pi / (sqrt(3) * alattice)
    b1 = np.array([sqrt(3) / 2 * G, -1 / 2 * G])
    b2 = np.array([sqrt(3) / 2 * G, 1 / 2 * G])
    akx = np.zeros((N, N))
    aky = np.zeros((N, N))

    kArr = np.zeros([N,N,2])
    for i in tqdm(range(1, N + 1), desc="Monkhorst-Pack k grid", colour="red"):
        for j in range(1, N + 1):

            kvec = (2 * i - N - 1) / (2 * N) * b1 + (2 * j - N - 1) / (2 * N) * b2
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    dkx = sqrt(3) / 2 * abs(akx[0, 0] - akx[1, 0])
    dky = 1 / 2 * abs(akx[0, 1] - akx[0, 0])

    kArr[:,:,0] = akx
    kArr[:,:,1] = aky
    return kArr, dkx, dky
