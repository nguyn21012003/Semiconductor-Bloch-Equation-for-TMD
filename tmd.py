import os

from core.parameters import getParams, paraNN, paraTNN
from function.bands import writeBandstructure
from function.mass import calcMass
from function.sbe import linearSBE
from settings.configs import Config
from utils.draw import drawBandstructure

cfg = Config()
base_dir = cfg.base_dir


class Material:
    def __init__(
        self, name: str, modelNeighbor: str, modelApprox: str, isMatrixTransform: bool
    ) -> None:
        self.name = name
        self.modelNeighbor = modelNeighbor  ## TNN or NN
        self.modelApprox = modelApprox  ## GGA or LDA
        self.isMatrixTransform = isMatrixTransform  # default is True

    def mass(self):
        data = getParams(self.name, self.modelNeighbor, self.modelApprox)
        self.alattice = data["alattice"]
        effMass = calcMass(data, self.modelNeighbor)
        self.mass_e, self.mass_h, self.mass_r = effMass

    """ This part recalculates the bandstructure taken from Ref Phys.rev.B 88,085433"""

    def bandstructure(self, fileMaterial):
        data = getParams(self.name, self.modelNeighbor, self.modelApprox)
        writeBandstructure(data, self.name, self.modelNeighbor, fileMaterial)

    def draw_bands(self, fileMaterial, fileDraw, dir):
        """This is optional, if u want to draw band then use this"""
        isDraw = True
        if isDraw:
            drawBandstructure(fileMaterial, fileDraw, dir, self.alattice, self.name)

    def absorptionSpectrum(self):
        data = getParams(self.name, self.modelNeighbor, self.modelApprox)
        linearSBE(data, self.modelNeighbor)


def main():
    isMatrixTransform = cfg.isMatrixTransform
    os.makedirs(base_dir, exist_ok=True)

    name = "MoS2"
    material = Material(name, cfg.neighbor, cfg.approx, isMatrixTransform)

    dir_path = f"{base_dir}/{name}"
    fileGnu = f"{dir_path}/drawBandStruct.gnuplot"
    fileMaterial = f"{dir_path}/bandstructure.dat"
    os.makedirs(dir_path, exist_ok=True)

    print(f"\n=== Processing {name} ===")
    ################################################# Effective mass without field
    # material.mass()
    # print(f"m_eff (hole):     {material.mass_h}")
    # print(f"m_eff (electron): {material.mass_e}")
    # print(f"m_eff (reduced):  {material.mass_r}")

    ################################################ Bandstructure
    # material.bandstructure(fileMaterial)
    # material.draw_bands(fileMaterial, fileGnu, base_dir)

    material.absorptionSpectrum()

    return None


if __name__ == "__main__":
    main()
