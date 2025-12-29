import numpy as np
from numpy import pi, sqrt
from tqdm import tqdm


def Rhombus(a0: float, N: int):
    G = 4 * pi / (sqrt(3) * a0)
    # ak1 = np.linspace(-G / 2, G / 2, N)
    # ak2 = np.linspace(-G / 2, G / 2, N)
    ak1 = np.zeros([N, N])
    ak2 = np.zeros([N, N])
    dkx = G / (N - 1)
    dky = G / (N - 1)
    for j in range(N):
        for i in range(N):
            ak1 = -G / 2 + (i - 1) * dkx
            ak2 = -G / 2 + (j - 1) * dky
    akx = np.zeros([N, N])
    aky = np.zeros([N, N])
    kArr = np.zeros([N, N, 2])
    for i in tqdm(range(N), desc="Create rhombus k grid", colour="red"):
        for j in range(N):
            akx[i][j] = sqrt(3) / 2 * (ak1[i] + ak2[j])
            aky[i][j] = -1 / 2 * (ak1[i] - ak2[j])
    kArr[:, :, 0] = akx
    kArr[:, :, 1] = aky

    return kArr, dkx, dky


def Monkhorst(a0: float, N: int):
    G = 4 * pi / (sqrt(3) * a0)
    b1 = np.array([sqrt(3) / 2 * G, -1 / 2 * G])
    b2 = np.array([sqrt(3) / 2 * G, 1 / 2 * G])
    kgrid = np.zeros([N, N, 2])
    ak = np.zeros(N)
    for i in range(N):
        ak[i] = (2.0 * (i + 1) - N - 1.0) / (2.0 * N)
    for i in tqdm(range(N), desc="Monkhorst-Pack k grid", colour="red"):
        for j in range(N):
            kgrid[i, j, 0] = ak[i] * b1[0] + ak[j] * b2[0]
            kgrid[i, j, 1] = ak[i] * b1[1] + ak[j] * b2[1]

    dkx = abs(kgrid[0, 0, 0] - kgrid[1, 0, 0])
    dky = abs(kgrid[0, 0, 1] - kgrid[0, 1, 1])
    print(dkx)
    print(dky)

    dk1 = np.sqrt(
        (kgrid[1, 0, 0] - kgrid[0, 0, 0]) ** 2 + (kgrid[1, 0, 1] - kgrid[0, 0, 1]) ** 2
    )
    dk2 = np.sqrt(
        (kgrid[0, 1, 0] - kgrid[0, 0, 0]) ** 2 + (kgrid[0, 1, 1] - kgrid[0, 0, 1]) ** 2
    )

    # delta_k = dkx * dky
    delta_k = dk1 * dk2

    return kgrid, delta_k


def Monkhorst0(a0: float, N: int):
    G = 4 * pi / (sqrt(3) * a0)
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

    kArr[:, :, 0] = akx
    kArr[:, :, 1] = aky
    # dkx = G / (N - 1)
    # dky = G / (N - 1)
    # dk2 = dkx * dky
    # print(dkx)
    # print(dky)

    areaBZ = abs(np.cross(b1, b2))
    dk2 = areaBZ / (N * N)

    return kArr, dk2


def Monkhorst_Gamma_Centered(a0: float, N: int):
    b1 = np.array([2 * pi / a0, -2 * pi / (sqrt(3) * a0)])
    b2 = np.array([0, 4 * pi / (sqrt(3) * a0)])
    b3 = np.array([-2 * pi / a0, -2 * pi / (sqrt(3) * a0)])
    grid = np.zeros((3, N, N, 2))

    akx = np.zeros((N, N))
    aky = np.zeros((N, N))

    ################# Rhombus 1
    kArr_1 = np.zeros((N, N, 2))
    for i in tqdm(range(1, N + 1), desc="MP rhombus 1", colour="red"):
        for j in range(1, N + 1):
            c1 = (i - 1) / N
            c2 = (j - 1) / N
            kvec = c1 * b1 + c2 * b2
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    kArr_1[:, :, 0] = akx
    kArr_1[:, :, 1] = aky

    ################# Rhombus 2
    kArr_2 = np.zeros((N, N, 2))
    for i in tqdm(range(1, N + 1), desc="MP rhombus 2", colour="green"):
        for j in range(1, N + 1):
            c1 = (i - 1) / N
            c2 = (j - 1) / N
            kvec = c1 * b1 + c2 * b3
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    kArr_2[:, :, 0] = akx
    kArr_2[:, :, 1] = aky

    ################# Rhombus 3
    kArr_3 = np.zeros((N, N, 2))
    for i in tqdm(range(1, N + 1), desc="MP rhombus 3", colour="blue"):
        for j in range(1, N + 1):
            c1 = (i - 1) / N
            c2 = (j - 1) / N
            kvec = c1 * b2 + c2 * b3
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    kArr_3[:, :, 0] = akx
    kArr_3[:, :, 1] = aky

    grid[0, :, :, :] = kArr_1
    grid[1, :, :, :] = kArr_2
    grid[2, :, :, :] = kArr_3

    areaBZ = abs(np.cross(b1, b2))
    dk2 = areaBZ / (N * N)

    return grid, dk2


def Monkhorst1(a0: float, N: int):
    b1 = np.array([2 * pi / a0, -2 * pi / (sqrt(3) * a0)])
    b2 = np.array([0, 4 * pi / (sqrt(3) * a0)])
    b3 = np.array([-2 * pi / a0, -2 * pi / (sqrt(3) * a0)])
    grid = np.zeros((3, N, N, 2))

    akx = np.zeros((N, N))
    aky = np.zeros((N, N))

    ################# Rhombus 1
    kArr_1 = np.zeros((N, N, 2))
    for i in tqdm(range(1, N + 1), desc="MP rhombus 1", colour="red"):
        for j in range(1, N + 1):
            c1 = (2 * i - N - 1) / (2 * N)
            c2 = (2 * j - N - 1) / (2 * N)
            kvec = c1 * b1 + c2 * b2
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    kArr_1[:, :, 0] = akx
    kArr_1[:, :, 1] = aky

    ################# Rhombus 2
    kArr_2 = np.zeros((N, N, 2))
    for i in tqdm(range(1, N + 1), desc="MP rhombus 2", colour="green"):
        for j in range(1, N + 1):
            c1 = (2 * i - N - 1) / (2 * N)
            c2 = (2 * j - N - 1) / (2 * N)
            kvec = c1 * b1 + c2 * b3
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    kArr_2[:, :, 0] = akx
    kArr_2[:, :, 1] = aky

    ################# Rhombus 3
    kArr_3 = np.zeros((N, N, 2))
    for i in tqdm(range(1, N + 1), desc="MP rhombus 3", colour="blue"):
        for j in range(1, N + 1):
            c1 = (2 * i - N - 1) / (2 * N)
            c2 = (2 * j - N - 1) / (2 * N)
            kvec = c1 * b2 + c2 * b3
            akx[i - 1, j - 1] = kvec[0]
            aky[i - 1, j - 1] = kvec[1]

    kArr_3[:, :, 0] = akx
    kArr_3[:, :, 1] = aky

    grid[0, :, :, :] = kArr_1
    grid[1, :, :, :] = kArr_2
    grid[2, :, :, :] = kArr_3

    areaBZ = abs(np.cross(b1, b2))
    dk2 = areaBZ / (N * N)

    return grid, dk2


# def Cartesian(a0: float, N: int):
#     SR3 = sqrt(3)
#     B = np.array(
#         [
#             [2.0 * pi / a0, 2.0 * pi / a0],
#             [2.0 * pi / (SR3 * a0), -2.0 * pi / (SR3 * a0)],
#         ]
#     )
#     ratio = np.linspace(0, 1, N)
#
#     nk1_ratio = ratio.reshape(-1, 1)
#     nk2_ratio = ratio.reshape(1, -1)
#     grid_x = (B[0, 0] * nk1_ratio + B[0, 1] * nk2_ratio) - 2.0 * pi / a0
#     grid_y = (B[1, 0] * nk1_ratio + B[1, 1] * nk2_ratio) - 0.0
#     kArr = np.stack((grid_x, grid_y), axis=-1)
#     k1max = (2.0 * pi) / (SR3 * a0)
#     k2max = (2.0 * pi) / (SR3 * a0)
#     dk1 = 2.0 * k1max / (N - 1)
#     dk2 = 2.0 * k2max / (N - 1)
#     k1 = -k1max + np.arange(N) * dk1
#     k2 = -k2max + np.arange(N) * dk2
#     term1_x = SR3 * (k1[0] + k2[0]) / 2.0
#     term2_x = SR3 * (k1[1] + k2[0]) / 2.0
#     dkx = abs(term1_x - term2_x)
#     term1_y = (k2[0] - k1[0]) / 2.0
#     term2_y = (k2[0] - k1[1]) / 2.0
#     dky = abs(term1_y - term2_y)
#
#     dS = dkx * dky
#     return kArr, dS
