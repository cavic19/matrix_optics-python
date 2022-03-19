from optix.matrixopt.ABCDformalism import ABCDElement, ABCDCompositeElement, Media
from optix.beams import GaussianBeam
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["OpticalPath", "Drawer"]

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
        # If output is in some kind of media, take the refractive index of it 
        refractive_index = self.childs[-1].n if isinstance(self.childs[-1], Media) else 1
        # TODO: Not true! Didnt acount for losses due to a reflection and so! (Maybe give ABCDElement rights to alter a GaussianBeam state, so you conform OCP..)
        amplitude = input.amplitude
        return GaussianBeam.from_q(wave_length=input.wavelength, q=q_out, z_pos=self.length, refractive_index=refractive_index, amplitude=amplitude)

    def __update_matrix(self):
        self.matrix = self._build_matrix()

class Drawer:
    __UNITS = {
        "um": 10**6,
        "mm": 10**3,
        "cm": 10**2,
        "dm": 10**1,
        "m":  10**0
    }


    def __init__(self, op: OpticalPath, gauss_in: GaussianBeam, **kwargs) -> None:
        self._elements = op.childs
        self.z_unit = kwargs.get("z_unit", "mm")
        self.w_unit = kwargs.get("w_unit", "mm")
        self.color = kwargs.get("color", "blue")
        self.draw_childs = kwargs.get("draw_childs", True)
        self._op_temp = OpticalPath()
        self._gauss_in = gauss_in
        self._fig, self._ax = plt.subplots()
        self._current_z = 0


    def show(self):
        self.draw().show()

    def draw(self):
        for i, element in enumerate(self._elements):
            self.draw_element(element)
            self._ax.axvline(self.__z_unit_transform(self._current_z) , linewidth=1, color="black")
            # self.ax.annotate(f" {i+1}.", (self.__z_unit_transform(self.current_z), 0))

            self._ax.set_xlabel(f"Distance [{self.z_unit}]")
            self._ax.set_ylabel(f"W [{self.w_unit}]")
        # self.ax.legend(self.__build_legend(),handletextpad=-2.0, handlelength=0)
        return self._fig      

    def draw_element(s, element):
        if isinstance(element, ABCDCompositeElement):
            if s.draw_childs:   
                for child in element.childs:
                    s.draw_element(child)
            else:
                assert False, "Not implemented!"
                s.__op_temp.append(element)
        elif isinstance(element, ABCDElement):
            s._op_temp.append(element)
            if element.length > 0:
                gauss_temp = s._op_temp.propagate(s._gauss_in)
                z = np.linspace(s._current_z, s._current_z + element.length, 100)
                w = gauss_temp.beam_radius(z)
                s._ax.plot(s.__z_unit_transform(z), s.__w_unit_transform(w), label=element.name, color=s.color)
                s._current_z += element.length

    def __z_unit_transform(self, z):
        return z * self.__UNITS[self.z_unit]

    def __w_unit_transform(self, w):
        return w * self.__UNITS[self.w_unit]
            
    def __build_legend(self):
        labels = [f"{i+1}. {e.name}" for i, e in enumerate(self._elements)]
        return labels