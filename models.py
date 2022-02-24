from typing import Any
from helper import count_matching

class GaussianBeam:
    """Gaussain beam in z direction. Not mutable."""
    
    __NOT_SIMULTANOUS_KWARGS = ["q", "W0", "z0", "z_waist", "divergence"]
    __DEFAULT_REFRACTIVE_INDEX = 1
    def __init__(self, amplitude, **kwargs) -> None:
        self.amplitude = amplitude  
        if count_matching(kwargs.keys(), lambda key: key in self.__NOT_SIMULTANOUS_KWARGS) != 2:
            raise ValueError(f"Just 2 of the key arguments {', '.join(self.__NOT_SIMULTANOUS_KWARGS)} can be provided.")       
        self.__n = self.__get_kwarg_if_present(kwargs, "n", self.__DEFAULT_REFRACTIVE_INDEX)     
        self.__args = {k: kwargs[k] for k in kwargs if k in self.__NOT_SIMULTANOUS_KWARGS}

    def __get_kwarg_if_present(self, kwargs: dict, kw: str, otherwise) -> Any:
        if kw in kwargs:
            return kwargs[kw]
        return otherwise

    @property
    def q(self):
        if "q" in self.__args:
            return self.__args["q"]
        assert False, "Not implemented"

    @property
    def W0(self):
        if "W0" in self.__args:
            return self.__args["W0"]
        assert False, "Not implemented"

    @property
    def z0(self):
        if "z0" in self.__args:
            return self.__args["z0"]
        assert False, "Not implemented"
        
    @property
    def z_waist(self):
        if "z_waist" in self.__args:
            return self.__args["z_waist"]
        assert False, "Not implemented"
    
    @property
    def divergence(self):
        if "divergence" in self.__args:
            return self.__args["divergence"]
        assert False, "Not implemented"

    def W(self, z: float) -> float:
        assert False, "Not implemented"

    def R(self, z: float) -> float:
        assert False, "Not implemented"
    
    def xi(self, z: float) -> float:
        assert False, "Not implemented"