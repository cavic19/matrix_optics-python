from ast import Assert
import unittest
import cmath
from cv2 import exp
import numpy as np
from optix.ABCDformalism import *

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
    def test_should_equal(self):
        q_in = complex(3,4)
        free_space = FreeSpace(10)
        q_out_expected = complex(13, 4)
        q_out_actual = free_space.act(q_in)
        self.assertEquals(q_out_actual, q_out_expected)

class TestPlanConvexLens(unittest.TestCase):
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

    



    

        

        