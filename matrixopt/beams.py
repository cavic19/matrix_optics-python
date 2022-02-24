from collections import namedtuple
from math import pi, sqrt

class GaussianBeam:
    _SUPPORTED_KWARGS = ["w0", "zr", "div"] #waist radius, rayleigh range, divergence

    def __init__(self, 
                wave_length, 
                amplitude = 1, 
                refractive_index = 1,
                waist_location = 0, 
                **beam_param) -> None:
        """
        Args:
            beam_param: Specifiy one of the following parameters: waist radius ("w0"), rayleigh range ("zr"), divergence ("div")

        Raises:
            ValueError: When no or more then 1 beam_param is presented.
        """
        self._wavelength = wave_length
        self._amplitude = amplitude
        self._refractive_index = refractive_index
        self._waist_location = waist_location
        if len(beam_param) != 1 or not list(beam_param)[0] in self._SUPPORTED_KWARGS:
            raise ValueError(f"One of {', '.join(self._SUPPORTED_KWARGS)} arguments must be presented!")    
        
        BeamParam = namedtuple("BeamParam", "name value")
        self.__beam_param = BeamParam(list(beam_param)[0], beam_param[list(beam_param)[0]])


    @property
    def divergence(self):
        if "div" == self.__beam_param.name:
            return self.__beam_param.value
        return self.wavelength / (pi * self.waist_radius * self.refractive_index)


    @property
    def waist_radius(self): 
        if "w0" == self.__beam_param.name:
            return self.__beam_param.value
        return sqrt((self.wavelength * self.rayleigh_range) / (pi * self.refractive_index))

    @property
    def rayleigh_range(self):
        if "zr" == self.__beam_param.name:
            return self.__beam_param.value
        return self.wavelength / (pi * self.refractive_index * self.divergence**2)

    @property
    def wavelength(self):
        return self._wavelength
    
    @property
    def refractive_index(self):
        return self._refractive_index

    def beam_radius(self, z) -> float:
        return self.waist_radius * sqrt(1 + ((z - self._waist_location) / self.rayleigh_range)**2)

    def curviture(self, z) -> float:
        return (z - self._waist_location) * (1 + (self.rayleigh_range / (z - self._waist_location)))

    def cbeam_parameter(self, z) -> complex:
        return complex(z- self._waist_location, self.rayleigh_range)