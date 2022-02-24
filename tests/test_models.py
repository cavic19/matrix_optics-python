import unittest
from matrixopt.models import *


class TestGaussianBeam(unittest.TestCase):
    TEST_VALUES = {
        "q": complex(121,589),
        "W0": 0.001,
        "z0": 0.123,
        "divergence": 10,
        "z_waist": 5,
        "n": 1
        }

    def test_initializer_success(self):
        gb = GaussianBeam(1, q=self.TEST_VALUES["q"], W0=self.TEST_VALUES["W0"], a=5)
        self.assertEquals(gb.q, self.TEST_VALUES["q"])
        self.assertEquals(gb.W0, self.TEST_VALUES["W0"])
        
        gb = GaussianBeam(1, z0=self.TEST_VALUES["z0"], z_waist=self.TEST_VALUES["z_waist"], a=5, b=6, c=7)
        self.assertEquals(gb.z0, self.TEST_VALUES["z0"])
        self.assertEquals(gb.z_waist, self.TEST_VALUES["z_waist"])

        gb = GaussianBeam(1, z0=self.TEST_VALUES["z0"], divergence=self.TEST_VALUES["divergence"])
        self.assertEquals(gb.divergence, self.TEST_VALUES["divergence"])


    def test_initializer_fail(self):
        with self.assertRaises(ValueError):
            gb = GaussianBeam(1, q=self.TEST_VALUES["q"])
        with self.assertRaises(ValueError):
            gb = GaussianBeam(1, z0=self.TEST_VALUES["z0"], divergence=self.TEST_VALUES["divergence"], q=self.TEST_VALUES["q"])


