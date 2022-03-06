from functools import reduce
from typing import List
import numpy as np

class ABCDElement:
    @property
    def length(self) -> float:
        return 0

    def __init__(self, *args, name="") -> None:
        self.name = name
        """Accepts A, B, C, D matrix elements or a matrix itself"""
        if len(args) == 4:
            self._A = args[0]
            self._B = args[1]
            self._C = args[2]
            self._D = args[3]
        elif len(args) == 1 and isinstance(args[0], np.ndarray) and self.__is_square_matrix_of_dim(args[0], 2):
            self.matrix = args[0]
        else:
            raise ValueError("No matrix definition present in init.")

    def __is_square_matrix_of_dim(self, m: np.ndarray, dim: int):
        return all(len(row) == len(m) for row in m) and len(m) == dim

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self._A, self._B], [self._C, self._D]])

    @matrix.setter
    def matrix(self, value: np.ndarray):
        self._A = value[0][0]
        self._B = value[0][1]
        self._C = value[1][0]
        self._D = value[1][1]
    
    def act(self, q_param: complex) -> complex:
        nom = self._A * q_param + self._B
        denom = self._C * q_param + self._D
        return nom / denom

class Media(ABCDElement):
    @property
    def length(self) -> float:
        return self._d
    def __init__(self, d, n):
        self._d = d
        self.n = n
        super().__init__(1, d, 0, 1, name=f"Media(d={d}, n={n})")


class FreeSpace(Media):
    """Propagation in free space or in a medium of constant refractive index"""
    @property
    def length(self) -> float:
        return self._d

    def __init__(self, d) -> None:
        self._d = d
        super().__init__(d=d, n=1)
        self.name = f"FreeSpace(d={d})"

class ThinLens(ABCDElement):
    """Thin lens aproximation. Only valid if the focal length is much greater than the thickness of the lens"""
    @property
    def f(self):
        return self._f

    def __init__(self, f: float) -> None:
        self._f = f
        super().__init__(1, 0, -1/f, 1, name=f"ThinLens(f={f})")


class FlatInterface(ABCDElement):
    """Refraction at a flat interface"""
    def __init__(self, n1, n2) -> None:
        """

        Args:
            n1 (float): Refractive index of first media
            n2 (float): Refractive index of second media
        """
        super().__init__(1, 0, 0, n1 / n2, name=f"FlatInterface(n1={n1}, n2={n2})")


class CurvedInterface(ABCDElement):
    """Refraction at a curved interface"""
    @property
    def n1(self):
        return self._n1

    @property
    def n2(self):
        return self._n2

    @property
    def R(self):
        return self._R

    def __init__(self, n1, n2, R) -> None:
        """
        Args:
            n1 (float): Refractive index of the material the ray is propagating from
            n2 (float): Refractive index of the material the ray is propagating to
            R (float): Curviture of the boundary that is positive for convex boundary and negative for concave boundary.
        """
        self._n1 = n1
        self._n2 = n2
        self._R = R
        super().__init__(self.__build_matrix(), name=f"CurvedInterface(n1={n1}, n2={n2}, R={R})")

    def __build_matrix(self) -> np.ndarray:
        return np.array([
            [1,                                             0],
            [-1*(self.n2 - self.n1) / (self.n2 * self.R),   self.n1 / self.n2]
        ])

class ABCDCompositeElement(ABCDElement):
    """Represents ABCDelement that consists of child elements"""
    @property
    def length(self) -> float:
        return reduce(lambda a, b: a +b , [e.length for e in self.childs])

    def __init__(self, childs: List[ABCDElement], name="") -> None:
        self.name = ""
        self.childs = childs
        super().__init__(self._build_matrix(), name=name)

    def _build_matrix(self) -> np.ndarray:
        if len(self.childs) == 0:
            return np.identity(2)
        return reduce(lambda c, b: c.dot(b), [e.matrix for e in reversed(self.childs)])

class ThickLens(ABCDCompositeElement):
    """Propagation through ThickLens."""
    @property
    def f(self) -> float:
        # Using Lens Maker's formula
        # + before 1/R2 is due to assumed positive R2
        f_inv =  (self._n/1 - 1) * (1/self._R1 + 1/self._R2) 
        return 1/f_inv

    def __init__(self, R1, n, R2, d) -> None:
        """ It is assumed, that the refractive index of free space is 1

        Args:
            R1 (float, positive): Curviture of the first face of the lens
            n (float): Refractive index of the lens
            R2 (float, positive): Curviture of the second face of the lens
            d (float): Thickness of the lens
        """
        self._n = n
        self._R1 = R1
        self._R2 = R2
        self._d = d

        components = [
            CurvedInterface(1, n, R1),
            Media(d, n),
            CurvedInterface(n, 1, -R2)
        ]

        super().__init__(components, name=f"ThickLens(R1={R1}, d={d}, R2={R2}, n={n})")





class PlanoConvexLens(ThickLens):
    def __init__(self, R, d, n) -> None:
        super().__init__(float("inf"), n, R, d)
        self.name = f"PlanConvexLens(R={R}, d={d}, n={n})"

