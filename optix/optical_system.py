from optix.ABCDformalism import ABCDElement
from optix.beams import GaussianBeam
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np

class OpticalPath:
    """Represents optical path that is created in init function."""
    def __init__(self, *elements: list[ABCDElement]) -> None:
        self._elements = list(elements)

    #TODO: Otestova funkci
    def append(self, element: ABCDElement) -> None:
        self._elements.append(element)

    def __len__(self) -> int:
        return len(self._elements)

    def propagate(self, input: GaussianBeam) -> GaussianBeam:
        q_in = input.cbeam_parameter(0)
        q_out = self.system.act(q_in)
        return GaussianBeam.from_q(input.wavelength, q_out, self.length, input.refractive_index, input.amplitude)
    
    @property
    def length(self):
        return reduce(lambda a, b: a +b , [e.length for e in self._elements])

    @property
    def system(self):
        return self.__build_system()

    def __build_system(self) -> ABCDElement:
        system_matrix = reduce(lambda c, b: c.dot(b), [e.matrix for e in reversed(self._elements)])
        return ABCDElement(system_matrix)

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