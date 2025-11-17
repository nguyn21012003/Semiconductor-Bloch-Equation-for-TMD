import numpy as np
from settings.configs import Config
import csv


def Et(t, E0, t_L):

    env = E0 * np.exp(-(t**2) / t_L**2)

    Ex = env
    Ey = 0

    return Ex, Ey


def At(w0):
    Axt, Ayt = 0.0, 0.0

    tmax = Config.tmax * 1e-15
    tmin = Config.tmin * 1e-15
    dt = Config.dt * 1e-15
    ntmax = int((tmax - tmin) / dt)

    with open("./results/pulse/pulse.dat", "w") as f:
        header = ["time", "Ex", "Ey", "Ax", "Ay"]
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for nt in range(1, ntmax + 1):

            time = tmin + nt * dt

            Ext, Eyt = Et(time, w0)

            Axt -= Ext * dt
            Ayt -= Eyt * dt
            row = {"time": time * 1e15, "Ex": Ext, "Ey": Eyt, "Ax": Axt, "Ay": Ayt}
            writer.writerow(row)
