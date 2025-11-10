from numpy import exp, sqrt

from core.irrMatrix import IR, IRNN, IRTNN
from core.irrMatrixTransform import IR as IR_tran
from core.irrMatrixTransform import IRNN as IRNN_tran
from core.irrMatrixTransform import IRTNN as IRTNN_tran


def tbm_Hamiltonian(alpha, beta, data, modelNeighbor, a):
    E0, h1, h2, h3, h4, h5, h6 = IR_tran(data)
    ham = (
        E0
        + exp(2j * alpha) * h1
        + exp(1j * (alpha - beta)) * h2
        + exp(1j * (-alpha - beta)) * h3
        + exp(-2j * alpha) * h4
        + exp(1j * (-alpha + beta)) * h5
        + exp(1j * (alpha + beta)) * h6
    )
    dhkx = (
        1j * a * exp(1j * 2 * alpha) * h1
        + 1j * a / 2 * exp(1j * (alpha - beta)) * h2
        - 1j * a / 2 * exp(-1j * (alpha + beta)) * h3
        - 1j * a * exp(-1j * 2 * alpha) * h4
        - 1j * a / 2 * exp(1j * (-alpha + beta)) * h5
        + 1j * a / 2 * exp(1j * (alpha + beta)) * h6
    )
    dhky = (
        1j * a * sqrt(3) / 2 * exp(1j * (alpha - beta)) * h2
        + 1j * a * sqrt(3) / 2 * exp(-1j * (alpha + beta)) * h3
        - 1j * a * sqrt(3) / 2 * exp(1j * (-alpha + beta)) * h5
        - 1j * a * sqrt(3) / 2 * exp(1j * (alpha + beta)) * h6
    )

    if modelNeighbor == "TNN":
        o1, o2, o3, o4, o5, o6 = IRTNN_tran(data)
        v1, v2, v3, v4, v5, v6 = IRNN_tran(data)

        ham += (
            +exp(4j * alpha) * o1
            + exp(2j * (alpha - beta)) * o2
            + exp(2j * (-alpha - beta)) * o3
            + exp(-4j * alpha) * o4
            + exp(2j * (-alpha + beta)) * o5
            + exp(2j * (alpha + beta)) * o6
            + exp(1j * (3 * alpha - beta)) * v1
            + exp(1j * (-2 * beta)) * v2
            + exp(1j * (-3 * alpha - beta)) * v3
            + exp(1j * (-3 * alpha + beta)) * v4
            + exp(1j * (2 * beta)) * v5
            + exp(1j * (3 * alpha + beta)) * v6
        )
        dhkx += (
            1j
            * a
            / 2
            * (
                4 * exp(4j * alpha) * o1
                + 2 * exp(2j * (alpha - beta)) * o2
                - 2 * exp(2j * (-alpha - beta)) * o3
                - 4 * exp(-4j * alpha) * o4
                - 2 * exp(2j * (-alpha + beta)) * o5
                + 2 * exp(2j * (alpha + beta)) * o6
                + 3 * exp(1j * (3 * alpha - beta)) * v1
                - 3 * exp(1j * (-3 * alpha - beta)) * v3
                - 3 * exp(1j * (-3 * alpha + beta)) * v4
                + 3 * exp(1j * (3 * alpha + beta)) * v6
            )
        )
        dhky += (
            1j
            * sqrt(3)
            * a
            / 2
            * (
                -2 * exp(2j * (alpha - beta)) * o2
                - 2 * exp(2j * (-alpha - beta)) * o3
                + 2 * exp(2j * (-alpha + beta)) * o5
                + 2 * exp(2j * (alpha + beta)) * o6
                - exp(1j * (3 * alpha - beta)) * v1
                - 2 * exp(1j * (-2 * beta)) * v2
                - exp(1j * (-3 * alpha - beta)) * v3
                + exp(1j * (-3 * alpha + beta)) * v4
                + 2 * exp(1j * (2 * beta)) * v5
                + exp(1j * (3 * alpha + beta)) * v6
            )
        )

    return ham,dhkx,dhky
