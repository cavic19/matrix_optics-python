import unittest
import cmath
from cv2 import exp
import numpy as np
from matrixopt.matrix_optics import ABCDElement, FreeSpace, OpticalPath

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

class TestThinLens(unittest.TestCase):
    pass


class TestOpticalPath(unittest.TestCase):
    def test_propagate(self):
        A = np.array([[0, 1], [2, 3]])
        B = np.array([[4, 5], [6, 7]])
        C = np.array([[8, 9], [8, 7]])
        q_in = complex(3, 5)
        # Vypočteno ručně
        expected = complex(1_803_075/1_557_725, -320/1_557_725)

        A_el = ABCDElement(A)
        B_el = ABCDElement(B)
        C_el = ABCDElement(C)

        op = OpticalPath(A_el, B_el, C_el)
        
        actual = op.propagate(q_in)
        self.assertAlmostEqual(expected, actual, places=3)

    def test_init_number(self):
        d1 = 5
        d2 = 7
        q_in = complex(3, 5)
        expected = q_in + d1 +d2
        op = OpticalPath(d1, d2)
        actual = op.propagate(q_in)
        self.assertEquals(actual, expected)

    def test_init_number_ABCDElement(self):
        d = 5
        A = np.array([[1, 2], [3, 4]])
        A_el = ABCDElement(A)

        q_in = complex(3, 5)
        expected = complex(355/1009, -10/1009)
        op = OpticalPath(d, A_el)
        actual = op.propagate(q_in)
        self.assertAlmostEqual(expected, actual, places=3)
        
        

        