from optix.ABCDformalism import ABCDElement, ABCDCompositeElement
from optix.beams import GaussianBeam
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np


class OpticalPath(ABCDCompositeElement):
    def __init__(self, *elements: ABCDElement, name="") -> None:
        super().__init__(list(elements), name=name)
    
    def append(self, element: ABCDElement) -> None:
        self.childs.append(element)

    def __len__(self) -> int:
        return len(self.childs)

    def propagate(self, input: GaussianBeam) -> GaussianBeam:
        self.__update_matrix()
        q_in = input.cbeam_parameter(0)
        q_out = self.act(q_in)
        return GaussianBeam.from_q(input.wavelength, q_out, self.length, input.refractive_index, input.amplitude)

    def __update_matrix(self):
        self.matrix = self._build_matrix()



def draw_path(op: OpticalPath, gauss_in: GaussianBeam) -> plt.Figure:
    fig, ax = plt.subplots()
    
    elements = op._elements
    op = OpticalPath()
    z0 = 0
    for element in elements:
        op.append(element)
        if op.length - z0 > 0:
            gauss_temp = op.propagate(gauss_in)
            z = np.linspace(z0, op.length, 100)
            w = gauss_temp.beam_radius(z)
            ax.plot(z, w, label=element.name)
            ax.axvline(op.length, linewidth=1, color="black")
            z0 = op.length
    return fig