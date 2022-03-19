from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
from optix.matrixopt import *
from optix.beams import GaussianBeam
from inspect import Parameter, signature, _empty
from scipy.optimize import minimize, shgo, basinhopping
import numpy as np
from decimal import Decimal
__all__ = ["Optimizer"]
ZERO_EPS = 1e-3
class Optimizer:
    INVALID_ELEMENT_TYPE_ERROR = "Valid arguments must by of type inheriting from type ABCDElement or instance of ABCDElement"

    elements: list = []
    def __init__(self, *elements, **kwargs):
        if any(not self._is_valid_element_type(e) for e in elements):
            raise ValueError(self.INVALID_ELEMENT_TYPE_ERROR)
        self.elements = list(elements)
    
    def append(self, element: Union[type, ABCDElement]) -> None:
        if not self._is_valid_element_type(element):
            raise ValueError(self.INVALID_ELEMENT_TYPE_ERROR)
        self.elements.append(element)

    def _is_valid_element_type(self, element: Any) -> bool:
        is_abcdelement_derived_type = isinstance(element, type) and issubclass(element, ABCDElement)
        is_abcdelement_instance = isinstance(element, ABCDElement)
        return is_abcdelement_derived_type or is_abcdelement_instance
    

    def run(self, 
            gauss_in: GaussianBeam, 
            optimal_w0: float, 
            optimal_w0_loc: float, 
            x0: List[tuple], 
            bounds=None,
            **kwargs) -> NamedTuple:
        param_num = self._get_param_number()
        if param_num == 0:
            raise ValueError("No parameters for optimization are present.")
        if len(x0) != param_num:
            raise ValueError(f"X0 mus be of the same length as the number of parameters ({param_num}).")

        bounds = bounds or [(ZERO_EPS, None)] * param_num
        fit_function = lambda params: self._distance_from_optimum(element_params=params, optimal_w0=optimal_w0, optimal_w0_loc=optimal_w0_loc, gauss_in=gauss_in)
        result = basinhopping(
            fit_function, 
            x0, 
            niter=kwargs.get("niter", 100), 
            minimizer_kwargs=dict(method="L-BFGS-B", bounds=bounds))
        return result.x
    
    def _distance_from_optimum(self, element_params, optimal_w0: float, optimal_w0_loc: float, gauss_in: GaussianBeam,) -> float:
        WEIGHT1 = 1000
        WEIGHT2 = 1
        actual_w0, actual_w0_loc = self._calculate_gauss_out(element_params, gauss_in)
        return (float(Decimal(actual_w0) * WEIGHT1 - Decimal(optimal_w0) * WEIGHT1)**2 + float(Decimal(actual_w0_loc) * WEIGHT2 - Decimal(optimal_w0_loc) * WEIGHT2)**2) / (WEIGHT1 + WEIGHT2)

    def _calculate_gauss_out(self, params, gauss_in: GaussianBeam) -> Tuple[float, float]:
        """For given gauss_in and set of params (for not concretized ABCDelements) calculates gauss_out

        Returns:
            (waist_radius, waist_location)
        """
        arg_index = 0
        op = OpticalPath()
        for e in self.elements:
            if isinstance(e, ABCDElement):
                op.append(e)
            else:
                arg_num = self._get_number_of_func_arguments(e.__init__) - 1 #because of self param
                op.append(e(*params[arg_index: arg_index + arg_num]))
                arg_index += arg_num
        gauss_out = op.propagate(gauss_in)
        return gauss_out.waist_radius, gauss_out.waist_location
    
    def _get_number_of_func_arguments(self, func):
        params = (p[1] for p in signature(func).parameters.items())
        non_optinal = (p for p in params if p.default == _empty)
        non_optional_positional = [p for p in non_optinal if p.kind == Parameter.POSITIONAL_OR_KEYWORD]
        return len(non_optional_positional) 

    def _get_param_number(self) -> int:
        type_elements = (e for e in self.elements if not isinstance(e, ABCDElement))
        type_elements_init_argument_numbers = (self._get_number_of_func_arguments(e.__init__) - 1 for e in type_elements)
        return sum(type_elements_init_argument_numbers)



