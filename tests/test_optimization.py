from multiprocessing.sharedctypes import Value
from pickletools import optimize
from random import gauss
import unittest
from optix.beams import GaussianBeam
from optix.optimization import *
from optix.matrixopt import *
class TestOptimizer(unittest.TestCase):
    def test_is_valid_element_type(self):
        Optimizer(FreeSpace(1))
        Optimizer(ABCDElement(1,2,3,4))
        Optimizer(ThinLens)
        Optimizer(ThinLens, FreeSpace)
        Optimizer(ThinLens, ABCDElement(1,2,3,4))
        with self.assertRaises(ValueError):
            Optimizer(float)
        with self.assertRaises(ValueError):
            Optimizer(123)
        with self.assertRaises(ValueError):
            Optimizer(ThinLens, 123)
        with self.assertRaises(ValueError):
            Optimizer(ThinLens(1), 123)          

    def test_get_number_of_init_arguments(self):
        def func(a, b, c, d=4, *args, **kwargs): pass
        expected = 3

        actual = Optimizer()._get_number_of_func_arguments(func)
        self.assertEquals(expected, actual)

    def test_run__no_degree_of_freedom_value_error(self):
        opt = Optimizer()
        opt.append(FreeSpace(10e-2))       
        opt.append(FreeSpace(1))                       
        opt.append(FlatInterface(1, 1.5))
        opt.append(FreeSpace(2e-2))
        opt.append(FlatInterface(1.5, 1))
        opt.append(ThinLens(1))
        gauss_in = GaussianBeam(405e-9, waist_location=-200, div=2e-3)
        with self.assertRaises(ValueError):
            opt.run(gauss_in, 1, 1, x0=[])


    def test_run_complex_system_1(self):
        test_path= OpticalPath()
        test_path.append(FreeSpace(10e-2))
        test_path.append(ThickLens(2e-2, 1.2, 5e-3, 10e-3))
        test_path.append(FreeSpace(42e-2))
        test_path.append(FlatInterface(1,1.5))
        test_path.append(FreeSpace(2e-2))
        test_path.append(FlatInterface(1.5, 1))
        test_path.append(ThinLens(500e-3))
        gauss_in = GaussianBeam(405e-9, waist_location=-200, div=2e-3)
        expected_gauss_out = test_path.propagate(gauss_in)

        class PartialyDefinedThickLens(ThickLens): 
            def __init__(self, R1): super().__init__(R1, 1.2, 5e-3, 10e-3)
        opt = Optimizer()
        opt.append(FreeSpace(10e-2))
        opt.append(PartialyDefinedThickLens)        # Unspecified R1
        opt.append(FreeSpace)                       # Unspecified d
        opt.append(FlatInterface(1, 1.5))
        opt.append(FreeSpace(2e-2))
        opt.append(FlatInterface(1.5, 1))
        opt.append(ThinLens)                        # Unspecified f

        R1, d, f = opt.run(
            gauss_in, 
            expected_gauss_out.waist_radius, 
            expected_gauss_out.waist_location, 
            x0=(1, 1, 1))

        test_output_path = OpticalPath()
        test_output_path.append(FreeSpace(10e-2))
        test_output_path.append(PartialyDefinedThickLens(R1))        # Unspecified R1
        test_output_path.append(FreeSpace(d))                       # Unspecified d
        test_output_path.append(FlatInterface(1, 1.5))
        test_output_path.append(FreeSpace(2e-2))
        test_output_path.append(FlatInterface(1.5, 1))
        test_output_path.append(ThinLens(f))
        acutal_gauss_out = test_output_path.propagate(gauss_in)     
        
        self.assertAlmostEqual(expected_gauss_out.waist_location, acutal_gauss_out.waist_location, delta=0.1e-3)
        self.assertAlmostEqual(expected_gauss_out.waist_radius, acutal_gauss_out.waist_radius, delta=0.1e-6)

    def test_run_complex_system_2(self):
        test_path= OpticalPath()
        test_path.append(FreeSpace(10e-2))
        test_path.append(PlanoConvexLens(2.1e-3, 1e-3, 1.7))
        test_path.append(FreeSpace(42e-2))
        test_path.append(ThinLens(200e-3))
        test_path.append(FreeSpace(60e-2))
        test_path.append(FlatInterface(1,1.5))
        test_path.append(FreeSpace(2e-2))
        test_path.append(FlatInterface(1.5, 1))
        gauss_in = GaussianBeam(2e-6, waist_location=65-2, div=10)
        expected_gauss_out = test_path.propagate(gauss_in)

        opt = Optimizer()
        opt.append(FreeSpace)               # <-- Unspecified (1 arg; l1)
        opt.append(PlanoConvexLens)               # <-- Unspecified (3 args; R, d, n)
        opt.append(FreeSpace(42e-2))
        opt.append(ThinLens(200e-3))
        opt.append(FreeSpace)               # <-- Unspecified (1 arg; l2)
        opt.append(FlatInterface(1,1.5))
        opt.append(FreeSpace(2e-2))         
        opt.append(FlatInterface(1.5, 1))

        l1, R, d, n, l2  = opt.run(
        gauss_in, 
        expected_gauss_out.waist_radius, 
        expected_gauss_out.waist_location, 
        x0=(1, 1, 1, 1, 1))

        test_output_path = OpticalPath()
        test_output_path.append(FreeSpace(l1))               # <-- Specified (1 arg; l1)
        test_output_path.append(PlanoConvexLens(R, d, n))               # <-- Specified (3 args; R, d, n)
        test_output_path.append(FreeSpace(42e-2))
        test_output_path.append(ThinLens(200e-3))
        test_output_path.append(FreeSpace(l2))               # <-- Specified (1 arg; l2)
        test_output_path.append(FlatInterface(1,1.5))
        test_output_path.append(FreeSpace(2e-2))         
        test_output_path.append(FlatInterface(1.5, 1))
        acutal_gauss_out = test_output_path.propagate(gauss_in)

        self.assertAlmostEqual(expected_gauss_out.waist_location, acutal_gauss_out.waist_location, delta=0.1e-3)
        self.assertAlmostEqual(expected_gauss_out.waist_radius, acutal_gauss_out.waist_radius, delta=0.1e-6)