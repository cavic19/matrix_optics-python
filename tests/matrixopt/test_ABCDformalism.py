from ast import Assert
import unittest
from cv2 import exp
import numpy as np
from optix.matrixopt import *

class TestABCDElement(unittest.TestCase):
    A = 1
    B = 2
    C = 3
    D = 4
    def test_init_matrix_elements_should_equal(s):
        el = ABCDElement(s.A, s.B, s.C, s.D)
        expected = np.array([[s.A, s.B], [s.C, s.D]])
        actual = el.matrix
        s.assertTrue((expected == actual).all())
    
    def test_init_matrix_should_equal(s):
        expected = np.array([[s.A, s.B], [s.C, s.D]])
        el = ABCDElement(expected)
        actual = el.matrix
        s.assertTrue((expected == actual).all())


    def test_init_invalid_args_should_raise(s):
        with s.assertRaises(ValueError):
            el = ABCDElement(1, 2, 3, 4, 5)
        with s.assertRaises(ValueError):
            el = ABCDElement(np.array([[1,2,3],[1,2,3]]))
        with s.assertRaises(ValueError):
            el = ABCDElement(np.array([[1,2],[1]]))

class TestFreeSpace(unittest.TestCase):
    def test_name(self):
        fs = FreeSpace(1.5)
        expected = "FreeSpace(d=1.5)"
        actual = fs.name
        self.assertEquals(expected, actual)

    def test_should_equal(self):
        q_in = complex(3,4)
        free_space = FreeSpace(10)
        q_out_expected = complex(13, 4)
        q_out_actual = free_space.act(q_in)
        self.assertEquals(q_out_actual, q_out_expected)

class TestThinLens(unittest.TestCase):
    def test_name(self):
        tl = ThinLens(1.5)
        expected = "ThinLens(f=1.5)"
        actual = tl.name
        self.assertEquals(expected, actual)

class TestFlatInterface(unittest.TestCase):
    def test_name(self):
        fi = FlatInterface(1.5,1)
        expected = "FlatInterface(n1=1.5, n2=1)"
        actual = fi.name
        self.assertEquals(expected, actual)

class TestCurvedInterface(unittest.TestCase):
    def test_name(self):
        ci = CurvedInterface(1.5, 1, 3)
        expected = "CurvedInterface(n1=1.5, n2=1, R=3)"
        actual = ci.name
        self.assertEquals(expected, actual)


class TestThickLens(unittest.TestCase):
    def test_name(self):
        tl = ThickLens(1,2,3, 4)
        expected = "ThickLens(R1=1, d=4, R2=3, n=2)"
        actual = tl.name
        self.assertEquals(expected, actual) 

class TestPlanConvexLens(unittest.TestCase):
    def test_name(self):
        pcl = PlanoConvexLens(1,2,3)
        expected = "PlanConvexLens(R=1, d=2, n=3)"
        actual = pcl.name
        self.assertEquals(expected, actual)

    def test_focal_length(self):
        lens = PlanoConvexLens(R=13.1e-3, n=1.5302, d=11.7e-3)
        expected = 24.7
        actual = round(lens.f * 10**3, 1)
        self.assertAlmostEquals(expected, actual, 2)

    def test_should_equal(self):
        """Modeling this example: https://dielslab.unm.edu/sites/default/files/lens_and_focusing_solutions.pdf"""
        n = 2
        d = 1
        R = 1
        expected_A = 1
        expected_B = 1/2
        expected_C = -1
        expected_D = 1/2
        
        pcl = PlanoConvexLens(R, d, n)
        actual_A = pcl._A
        actual_B = pcl._B
        actual_C = pcl._C
        actual_D = pcl._D

        self.assertEquals(actual_A, expected_A)
        self.assertEquals(actual_B, expected_B)
        self.assertEquals(actual_C, expected_C)
        self.assertEquals(actual_D, expected_D)

    



    

        

        