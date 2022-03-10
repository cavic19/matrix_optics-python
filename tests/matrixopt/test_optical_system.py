import unittest
from optix.matrixopt import *
import numpy as np
from optix.beams import GaussianBeam

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

    def test_free_space_thin_lens(self):
        d = 5
        f = 12
        dummy_input = GaussianBeam(405e-9,zr=0.5e-3)
        test_abcd = ABCDElement(np.array([[1, d], [-1/f , 1 - d/f]]))
        expected = test_abcd.act(dummy_input.cbeam_parameter(0))

        op = OpticalPath(FreeSpace(d), ThinLens(f))
        actual = op.propagate(dummy_input).cbeam_parameter(op.length)

        self.assertAlmostEquals(expected, actual)

    # def test_propagation_in_complex_system(self):
    #     """Inspiration: http://www.sfu.ca/~gchapman/e894/e894l6h.pdf"""
    #     d1 = 24
    #     f1 = 8
    #     d2 = 6
    #     f2 = -12
    #     d3 = 12
    #     expected = np.array([[25 - 0.1042 * d3, 12 - d3], [-0.1042, -1]])

    #     op = OpticalPath()
    #     op.append(FreeSpace(d1))
    #     op.append(FreeSpace(f1))
    #     op.append(FreeSpace(d2))
    #     op.append(FreeSpace(f2))
    #     op.append(FreeSpace(d3))
    #     actual = op.system.matrix

    #     np.testing.assert_array_equal(actual, expected)


    def test_propagation_in_complex_system2(self):
        """Comparing with the results from Olomouc (Czech Republic)"""
        gauss_in = GaussianBeam(400e-6, w0=1)
        op = OpticalPath()
        expected = 0.00636607

        op.append(ThinLens(50))
        op.append(FreeSpace(49.998))

        actual = op.propagate(gauss_in).waist_radius
        self.assertAlmostEquals(actual, expected,4)
        