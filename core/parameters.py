def paraNN(material: str, modelApprox: str) -> dict[str,float]:
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
    choice = matt.index(material)
    dataModel = {
        "LDA": {
            "alattice": [3.129, 3.132, 3.254, 3.253, 3.472, 3.476],
            "e1": [1.238, 1.355, 1.001, 1.124, 0.618, 0.623],
            "e2": [2.366, 2.569, 2.239, 2.447, 2.126, 2.251],
            "t0": [-0.218, -0.238, -0.222, -0.242, -0.202, -0.209],
            "t1": [0.444, 0.626, 0.350, 0.506, 0.254, 0.388],
            "t2": [0.533, 0.557, 0.488, 0.514, 0.423, 0.442],
            "t11": [0.250, 0.324, 0.244, 0.305, 0.241, 0.272],
            "t12": [0.360, 0.405, 0.314, 0.353, 0.263, 0.295],
            "t22": [0.047, -0.076, 0.129, 0.025, 0.269, 0.200],
        },
        "GGA": {
            "alattice": [3.190, 3.191, 3.326, 3.325, 3.357, 3.560],
            "e1": [1.046, 1.130, 0.919, 0.943, 0.605, 0.606],
            "e2": [2.104, 2.275, 2.065, 2.179, 1.972, 2.102],
            "t0": [-0.184, -0.206, -0.188, -0.207, -0.169, -0.175],
            "t1": [0.401, 0.567, 0.317, 0.457, 0.228, 0.342],
            "t2": [0.507, 0.536, 0.456, 0.486, 0.390, 0.410],
            "t11": [0.218, 0.286, 0.211, 0.263, 0.207, 0.233],
            "t12": [0.338, 0.384, 0.290, 0.329, 0.239, 0.270],
            "t22": [0.057, -0.061, 0.130, 0.034, 0.252, 0.190],
        },
    }
    e1 = dataModel[modelApprox]["e1"][choice]
    e2 = dataModel[modelApprox]["e2"][choice]
    t0 = dataModel[modelApprox]["t0"][choice]
    t1 = dataModel[modelApprox]["t1"][choice]
    t2 = dataModel[modelApprox]["t2"][choice]
    t11 = dataModel[modelApprox]["t11"][choice]
    t12 = dataModel[modelApprox]["t12"][choice]
    t22 = dataModel[modelApprox]["t22"][choice]
    alattice = dataModel[modelApprox]["alattice"][choice]

    data = {
        "material": matt[choice],
        "alattice": alattice,
        "e1": e1,
        "e2": e2,
        "t0": t0,
        "t1": t1,
        "t2": t2,
        "t11": t11,
        "t12": t12,
        "t22": t22,
    }

    return data


def paraTNN(material: str, modelApprox: str) -> dict[str,float]:
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
    choice = matt.index(material)

    dataModel = {
        "LDA": {
            "alattice": [3.129, 3.132, 3.254, 3.253, 3.472, 3.476],
            "e1": [0.820, 0.905, 0.715, 0.860, 0.574, 0.675],
            "e2": [1.931, 2.167, 1.687, 1.892, 1.410, 1.489],
            "t0": [-0.176, -0.175, -0.154, -0.152, -0.148, -0.124],
            "t1": [-0.101, -0.090, -0.134, -0.125, -0.173, -0.159],
            "t2": [0.531, 0.611, 0.437, 0.508, 0.333, 0.362],
            "t11": [0.084, 0.043, 0.124, 0.094, 0.203, 0.196],
            "t12": [0.169, 0.181, 0.119, 0.129, 0.186, 0.101],
            "t22": [0.070, 0.008, 0.072, 0.009, 0.127, 0.044],
            "r0": [0.070, 0.075, 0.048, 0.044, 0.007, -0.009],
            "r1": [-0.252, -0.282, -0.248, -0.278, -0.280, -0.25],
            "r2": [0.084, 0.127, 0.090, 0.129, -0.067, 0.129],
            "r11": [0.019, 0.001, 0.066, 0.059, 0.073, 0.131],
            "r12": [0.093, 0.114, 0.045, 0.058, 0.081, -0.007],
            "u0": [-0.043, -0.063, -0.067, -0.090, -0.054, -0.086],
            "u1": [0.047, 0.047, 0.041, 0.039, 0.008, 0.012],
            "u2": [0.005, 0.004, 0.005, 0.001, 0.037, -0.020],
            "u11": [0.304, 0.374, 0.327, 0.392, 0.145, 0.361],
            "u12": [-0.192, -0.224, -0.194, -0.224, -0.078, -0.193],
            "u22": [-0.162, -0.177, -0.151, -0.165, 0.035, -0.129],
        },
        "GGA": {
            "alattice": [3.190, 3.191, 3.326, 3.325, 3.357, 3.560],
            "e1": [0.683, 0.717, 0.684, 0.728, 0.588, 0.697],
            "e2": [1.707, 1.916, 1.546, 1.655, 1.303, 1.380],
            "t0": [-0.146, -0.152, -0.146, -0.146, -0.226, -0.109],
            "t1": [-0.114, -0.097, -0.130, -0.124, -0.234, -0.164],
            "t2": [0.506, 0.590, 0.432, 0.507, 0.036, 0.368],
            "t11": [0.085, 0.047, 0.144, 0.117, 0.400, 0.204],
            "t12": [0.162, 0.178, 0.117, 0.127, 0.098, 0.093],
            "t22": [0.073, 0.016, 0.075, 0.015, 0.017, 0.038],
            "r0": [0.060, 0.069, 0.039, 0.036, 0.003, -0.015],
            "r1": [-0.236, -0.261, -0.209, -0.234, -0.025, -0.209],
            "r2": [0.067, 0.107, 0.069, 0.107, -0.169, 0.107],
            "r11": [0.016, -0.003, 0.052, 0.044, 0.082, 0.115],
            "r12": [0.087, 0.109, 0.060, 0.075, 0.051, 0.009],
            "u0": [-0.038, -0.054, -0.042, -0.061, 0.057, -0.066],
            "u1": [0.046, 0.045, 0.036, 0.032, 0.103, 0.011],
            "u2": [0.001, 0.002, 0.008, 0.007, 0.187, -0.013],
            "u11": [0.266, 0.325, 0.272, 0.329, -0.045, 0.312],
            "u12": [-0.176, -0.206, -0.172, -0.202, -0.141, -0.177],
            "u22": [-0.150, -0.163, -0.150, -0.164, 0.087, -0.132],
        },
    }
    e1 = dataModel[modelApprox]["e1"][choice]
    e2 = dataModel[modelApprox]["e2"][choice]
    t0 = dataModel[modelApprox]["t0"][choice]
    t1 = dataModel[modelApprox]["t1"][choice]
    t2 = dataModel[modelApprox]["t2"][choice]
    t11 = dataModel[modelApprox]["t11"][choice]
    t12 = dataModel[modelApprox]["t12"][choice]
    t22 = dataModel[modelApprox]["t22"][choice]
    r0 = dataModel[modelApprox]["r0"][choice]
    r1 = dataModel[modelApprox]["r1"][choice]
    r2 = dataModel[modelApprox]["r2"][choice]
    r11 = dataModel[modelApprox]["r11"][choice]
    r12 = dataModel[modelApprox]["r12"][choice]
    u0 = dataModel[modelApprox]["u0"][choice]
    u1 = dataModel[modelApprox]["u1"][choice]
    u2 = dataModel[modelApprox]["u2"][choice]
    u11 = dataModel[modelApprox]["u11"][choice]
    u12 = dataModel[modelApprox]["u12"][choice]
    u22 = dataModel[modelApprox]["u22"][choice]
    a = dataModel[modelApprox]["alattice"][choice]

    data = {
        "material": matt[choice],
        "alattice": a,
        "e1": e1,
        "e2": e2,
        "t0": t0,
        "t1": t1,
        "t2": t2,
        "t11": t11,
        "t12": t12,
        "t22": t22,
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "r11": r11,
        "r12": r12,
        "u0": u0,
        "u1": u1,
        "u2": u2,
        "u11": u11,
        "u12": u12,
        "u22": u22,
    }

    return data


def getParams(nameMaterial: str, modelNeighbor: str, modelApprox: str) -> dict[str, float]:

    data = {}
    if modelNeighbor == "NN":
        data = paraNN(nameMaterial, modelApprox)
    elif modelNeighbor == "TNN":
        data = paraTNN(nameMaterial, modelApprox)

    return data
