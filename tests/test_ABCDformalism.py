import unittest
import cmath
from cv2 import exp
import numpy as np
from matrixopt.ABCDformalism import *

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
        # Arrange
        A_el = ABCDElement(np.array([[0, 1], [2, 3]]))
        B_el = ABCDElement(np.array([[4, 5], [6, 7]]))
        C_el = ABCDElement(np.array([[8, 9], [8, 7]]))
        q_in = complex(3, 5)
        gauss_in = GaussianBeam.from_q(1, q_in, 0)
        # Vypočteno ručně
        expected_q = GaussianBeam.from_q(1, complex(1_803_075/1_557_725, -320/1_557_725), 0).cbeam_parameter(0)

        # Act
        op = OpticalPath()
        op.append(A_el)
        op.append(B_el)
        op.append(C_el)
        actual_q = op.propagate(gauss_in).cbeam_parameter(op.length)
        
        # Assert
        self.assertAlmostEqual(actual_q, expected_q, places=3)

    def test_propagate_in_free_space(self):
        # Arrange
        q_in = complex(3, 5)
        gauss_in = GaussianBeam.from_q(1,  q_in, 0)
        d1 = 5
        d2 = 7
        expected_q = complex(3, 5) + d1 + d2 #Freespace only adds constant real numbers

        # Act
        op = OpticalPath(FreeSpace(d1))
        op.append(FreeSpace(d2))
        actual_q= op.propagate(gauss_in).cbeam_parameter(op.length) #Checking complex beam parameter in a op.length distance

        # Assert
        self.assertEquals(actual_q, expected_q)

    def test_init_number_ABCDElement(self):
        # Arrange
        fs = FreeSpace(5)
        A_el = ABCDElement(np.array([[1, 2], [3, 4]]))
        gauss_in = GaussianBeam.from_q(1, complex(3, 5), 0)
        expected_q = complex(355/1009, -10/1009)

        # Act
        op = OpticalPath(fs, A_el)
        actual_q = op.propagate(gauss_in).cbeam_parameter(op.length)

        # Assert
        self.assertAlmostEqual(actual_q, expected_q, places=3)

    def test_length(self):
        op = OpticalPath(FreeSpace(5), FreeSpace(10), FreeSpace(1.3), FreeSpace(0.001))
        self.assertAlmostEquals(16.301, op.length, 5)

        op = OpticalPath(FreeSpace(1), ThinLens(1), FreeSpace(0.5))
        self.assertEquals(1.5, op.length)      
        

    def test_elements_length(self):
        op = OpticalPath(FreeSpace(5), FreeSpace(10), FreeSpace(1.3))
        op.append(FreeSpace(1))
        op.append(FreeSpace(1))
        expected = 5

        actual = len(op)
        self.assertEquals(expected, actual)

        

        