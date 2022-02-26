from ctypes import Union
import numpy as np
from functools import reduce
from matrixopt.beams import GaussianBeam

class ABCDElement:
    length = 0
    def __init__(self, *args) -> None:
        """Accepts A, B, C, D matrix elements or a matrix itself"""
        if len(args) == 4:
            self._A = args[0]
            self._B = args[1]
            self._C = args[2]
            self._D = args[3]
        elif len(args) == 1 and isinstance(args[0], np.ndarray) and self.__is_square_matrix_of_dim(args[0], 2):
            self._A = args[0][0][0]
            self._B = args[0][0][1]
            self._C = args[0][1][0]
            self._D = args[0][1][1]
        else:
            raise ValueError("No matrix definition present in init.")

    def __is_square_matrix_of_dim(self, m: np.ndarray, dim: int):
        return all(len(row) == len(m) for row in m) and len(m) == dim

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self._A, self._B], [self._C, self._D]])
    
    def act(self, q_param: complex) -> complex:
        nom = self._A * q_param + self._B
        denom = self._C * q_param + self._D
        return nom / denom


class FreeSpace(ABCDElement):
    def __init__(self, d: float, refr_index: float = 1) -> None:
        self.length = d
        self.n = refr_index
        super().__init__(1, d / refr_index, 0, 1)
  

class ThinLens(ABCDElement):
    def __init__(self, f: float) -> None:
        self.f = f
        super().__init__(1, 0, -1/f, 1)


class OpticalPath:
    """Represents optical path that is created in init function."""
    def __init__(self, *elements: list[ABCDElement]) -> None:
        self.elements = list(elements)

    def propagate(self, input: GaussianBeam) -> GaussianBeam:
        q_in = input.cbeam_parameter(0)
        system = self.__build_system()
        q_out = system.act(q_in)
        return GaussianBeam.from_q(input.wavelength, q_out, self.length, input.refractive_index, input.amplitude)

    def __build_system(self) -> ABCDElement:
        system_matrix = reduce(lambda c, b: c.dot(b), [e.matrix for e in reversed(self.elements)])
        return ABCDElement(system_matrix)
    
    @property
    def length(self):
        return reduce(lambda a, b: a +b , [e.length for e in self.elements])

