from __future__ import annotations
from collections import namedtuple
import math
import numpy as np
from typing import List, Union

def _process_images(positions, data: List[np.ndarray], wave_length, pixel_size, **kwargs) -> GaussianBeam:
    """Far field aproximation"""
    amplitudes = []
    radiuses = []
    for dataInput in data:
        ampl, rad = _extract_beam_characteristics(dataInput)
        amplitudes.append(ampl)
        radiuses.append(rad * pixel_size)
    divergence, w_loc = _calculate_div_and_waist_location(positions, radiuses)
    return GaussianBeam(
        wave_length=wave_length, 
        amplitude=kwargs.get("amplitude", 1), 
        refractive_index=kwargs.get("n",1), 
        waist_location=w_loc,
        divergence=divergence)

def _extract_beam_characteristics(data: np.ndarray) -> Union[float, float]:
    amplitude = np.amax(data)
    THRESHOLD = int(amplitude / np.e**2)
    Y_MAX, X_MAX = np.unravel_index(data.argmax(), data.shape)
    TOLERANCE = 1
    is_border_point = lambda a: abs(a - THRESHOLD) <= TOLERANCE
    get_radius = lambda x, y: np.sqrt((X_MAX - x)**2 + (Y_MAX - y)**2)
    radiuses = []
    for y in range(len(data)):
        for x in range(len(data[0])):
            if is_border_point(data[y][x]):
                radiuses.append(get_radius(x, y))
    return amplitude, np.average(radiuses)     

def _calculate_div_and_waist_location(positions, beam_radiuses) -> Union[float, float]:
    a, b = np.polyfit(positions, beam_radiuses, deg=1)
    return np.arctan(a), -b/a


class GaussianBeam:
    _SUPPORTED_KWARGS = ["w0", "zr", "div"] #waist radius, rayleigh range, divergence
    
    @property
    def divergence(self):
        if "div" == self.__beam_param.name:
            return self.__beam_param.value
        return self.wavelength / (math.pi * self.waist_radius * self.refractive_index)


    @property
    def waist_radius(self): 
        if "w0" == self.__beam_param.name:
            return self.__beam_param.value
        return math.sqrt((self.wavelength * self.rayleigh_range) / (math.pi * self.refractive_index))

    @property
    def rayleigh_range(self):
        if "zr" == self.__beam_param.name:
            return self.__beam_param.value
        return self.wavelength / (math.pi * self.refractive_index * self.divergence**2)

    @property
    def wavelength(self):
        return self._wavelength
    
    @property
    def refractive_index(self):
        return self._refractive_index
    
    @property
    def amplitude(self):
        return self._amplitude
    
    @property
    def waist_location(self):
        return self._waist_location
    
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
            ValueError: When no or more then 1 beam parameter is presented.
        """
        self._wavelength = wave_length
        self._amplitude = amplitude
        self._refractive_index = refractive_index
        self._waist_location = waist_location
        if len(beam_param) != 1 or not list(beam_param)[0] in self._SUPPORTED_KWARGS:
            raise ValueError(f"One of {', '.join(self._SUPPORTED_KWARGS)} arguments must be presented!")    
        
        BeamParam = namedtuple("BeamParam", "name value")
        self.__beam_param = BeamParam(list(beam_param)[0], beam_param[list(beam_param)[0]])

    @staticmethod
    def from_images(wave_length, positions, pixel_size, data, **kwargs) -> GaussianBeam:
        return _process_images(positions, data, wave_length, pixel_size, **kwargs)

    @staticmethod
    def from_q(wave_length, q: complex, z_pos: float, refractive_index=1, amplitude=1) -> GaussianBeam:
        """Creates a GaussianBeam instance from complex beam parameter "q".

        Args:
            q (complex): complex beam parameter
            z_pos (float): position at which the complex beam parameter has been evalaueted
        """
        waist_location = z_pos - q.real
        rayleigh_range = q.imag
        return GaussianBeam(wave_length, amplitude, refractive_index, waist_location, zr=rayleigh_range)

    def __str__(self) -> str:
        warning = ""
        if self.refractive_index != 1:
            warning = f"PROPAGATING IN MEDIA OF REFRACTIVE INDEX {self.refractive_index}!\n"
        return warning + f"""
        amplitude\t=\t{self.amplitude},
        wavelength\t=\t{self.wavelength*10**9} nm,
        waist_loc\t=\t{self.waist_location*10**2} cm,
        waist_rad\t=\t{self.waist_radius*10**3} mm,
        rayleigh_r\t=\t{self.rayleigh_range*10**3} mm,
        divergence\t=\t{self.divergence*10**3} mrad
        """

    def beam_radius(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.waist_radius * np.sqrt(1 + ((z - self.waist_location) / self.rayleigh_range)**2)

    def curviture(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (z - self.waist_location) * (1 + (self.rayleigh_range / (z - self.waist_location))**2)

    def cbeam_parameter(self, z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
        if isinstance(z, np.ndarray):
            return np.array([self.cbeam_parameter(i) for i in z])
        return complex(z- self.waist_location, self.rayleigh_range)