from typing import List, Tuple, Union, Iterable
from scipy.optimize import curve_fit
from optix.beams.beams import GaussianBeam
import numpy as np

__all__ = ["extract_gaussian_beam", "extract_m2"]

def extract_gaussian_beam(
    positions: np.ndarray, 
    profiles: Iterable[np.ndarray], 
    wave_length, 
    pixel_size, 
    **kwargs) -> GaussianBeam:
    """From image data and its position extract the gaussian beam
    Args:
        positions (np.ndarray): positions of taken image data (arbitrary coordinate system)
        data (Iterable[np.ndarray]): image data matching (sorted in the same fassion as positions array is)
        wave_length (_type_): wave_length of the gaussian beam
        pixel_size (_type_): pixel size of the camera that took the images (for unit preserving)
    Returns:
        GaussianBeam
    """
    amplitudes, radiuses = _extract_amplitudes_radiuses(profiles, pixel_size)
    divergence, w_loc = _calculate_div_waist_locations(positions, radiuses)
    return GaussianBeam(
        wave_length=wave_length, 
        amplitude=kwargs.get("amplitude", 1), 
        refractive_index=kwargs.get("n",1), 
        waist_location=w_loc,
        divergence=divergence)

def _calculate_div_waist_locations(positions, beam_radiuses) -> Union[float, float]:
    a, b = np.polyfit(positions, beam_radiuses, deg=1)
    div = np.arctan(a)
    w_loc = -b/a
    return div, w_loc



def extract_m2(positions: np.ndarray, profiles: Iterable[np.ndarray], pixel_size, wave_length, **kwargs) ->Tuple[GaussianBeam, float]:
    #TODO: Trigger warning when the positions are picked badly 
    # (they should obbey the rule that you pick ones in far field and near field as well)
    amplitudes, radiuses = _extract_amplitudes_radiuses(profiles, pixel_size)
    radiuses_squared = (r**2 for r in radiuses)
    def function(z, w0, M, w_loc):
        return w0**2 + M**4 * (wave_length / (np.pi * w0))**2 * (z - w_loc)**2
    popt, _ = curve_fit(function, positions, radiuses_squared)
    w0, M, w_loc = popt
    gb = GaussianBeam(
        wave_length=wave_length,
        amplitude=kwargs.get("amplitude", 1),
        refractive_index=kwargs.get("refractive_index", 1),
        waist_location=w_loc,
        w0=w0)
    return gb, M**2



def _extract_amplitudes_radiuses(profiles: Iterable[np.ndarray], pixel_size) -> Tuple[np.ndarray, np.ndarray]:
    amplitudes = []
    radiuses = []
    for profile in profiles:
        ampl, radius = _extract_amplitude_radius(profile)
        amplitudes.append(ampl)
        radiuses.append(radius*pixel_size)
    return np.array(amplitudes), np.array(radiuses)
        
TOLERANCE = 1
def _extract_amplitude_radius(profile: np.ndarray) -> Union[float, float]:
    amplitude = np.amax(profile)
    THRESHOLD = int(amplitude / np.e**2)
    Y_MAX, X_MAX = np.unravel_index(profile.argmax(), profile.shape)
    is_border_point = lambda a: abs(a - THRESHOLD) <= TOLERANCE
    get_radius = lambda x, y: np.sqrt((X_MAX - x)**2 + (Y_MAX - y)**2)
    radiuses = []
    for y in range(len(profile)):
        for x in range(len(profile[0])):
            if is_border_point(profile[y][x]):
                radiuses.append(get_radius(x, y))
    return amplitude, np.average(radiuses)     



