import unittest
from matrixopt.beams import *


class TestGaussianBeam(unittest.TestCase):

    # Values bellow are calculated here https://www.edmundoptics.com/knowledge-center/tech-tools/gaussian-beams/
    # TODO: Make input in appropriate units
    AMPLITUDE_TEST_VAL = 1
    W0_TEST_VAL = 1.2e-3 #1.2mm
    Z0_TEST_VAL = 11170.10721e-3 #11170.10721mm
    DIVERGENCE_TEST_VAL = 0.10743e-3 #0.10743mrad
    Z_WAIST_TEST_VAL = 0

    REFR_INDEX_TEST_VAL = 1
    WAVELNEGTH_TEST_VAL = 405e-9 #405nm



    def test_initializer_success(s):
        gb = GaussianBeam(s.AMPLITUDE_TEST_VAL, s.WAVELNEGTH_TEST_VAL,
        divergence=s.DIVERGENCE_TEST_VAL, 
        W0=s.W0_TEST_VAL, a=5)
        s.assertEquals(gb.divergence, s.DIVERGENCE_TEST_VAL)
        s.assertEquals(gb.W0, s.W0_TEST_VAL)
        
        gb = GaussianBeam(s.AMPLITUDE_TEST_VAL, s.WAVELNEGTH_TEST_VAL,
        z0=s.Z0_TEST_VAL, 
        z_waist=s.Z_WAIST_TEST_VAL, a=5, b=6, c=7)
        s.assertEquals(gb.z0, s.Z0_TEST_VAL)
        s.assertEquals(gb.z_waist, s.Z_WAIST_TEST_VAL)

        gb = GaussianBeam(s.AMPLITUDE_TEST_VAL, s.WAVELNEGTH_TEST_VAL,
        z0=s.Z0_TEST_VAL, 
        divergence=s.DIVERGENCE_TEST_VAL)
        s.assertEquals(gb.divergence, s.DIVERGENCE_TEST_VAL)

        # Takes only first 2 args and cuts the leftovers => z_waist shouldnt equal
        gb = GaussianBeam(s.AMPLITUDE_TEST_VAL, s.WAVELNEGTH_TEST_VAL, 
        z0=s.Z0_TEST_VAL, 
        divergence=s.DIVERGENCE_TEST_VAL, 
        z_waist=158)
        s.assertEquals(gb.z0, s.Z0_TEST_VAL)
        s.assertEquals(gb.divergence, s.DIVERGENCE_TEST_VAL)
        s.assertNotEquals(gb.z_waist, 158)


    def test_initializer_fail(self):
        with self.assertRaises(ValueError):
            gb = GaussianBeam(1, self.WAVELNEGTH_TEST_VAL, divergence=self.DIVERGENCE_TEST_VAL)
        with self.assertRaises(ValueError):
            gb = GaussianBeam(1, self.WAVELNEGTH_TEST_VAL)

    
    def test_parameters_recalculations(s):
        gb = GaussianBeam(
            s.AMPLITUDE_TEST_VAL, 
            s.WAVELNEGTH_TEST_VAL, 
            z0=s.Z0_TEST_VAL, 
            W0=s.W0_TEST_VAL)
        s.assertEquals(gb.divergence, s.DIVERGENCE_TEST_VAL)
        s.assertEquals(gb.z_waist, s.Z_WAIST_TEST_VAL)

        gb = GaussianBeam(
            s.AMPLITUDE_TEST_VAL, 
            s.WAVELNEGTH_TEST_VAL, 
            z0=s.Z0_TEST_VAL, 
            z_waist=s.Z_WAIST_TEST_VAL)
        s.assertEquals(gb.divergence, s.DIVERGENCE_TEST_VAL)
        s.assertEquals(gb.W0, s.W0_TEST_VAL)

        gb = GaussianBeam(
            s.AMPLITUDE_TEST_VAL, 
            s.WAVELNEGTH_TEST_VAL, 
            W0=s.W0_TEST_VAL, 
            z_waist=s.Z_WAIST_TEST_VAL)
        s.assertEquals(gb.divergence, s.DIVERGENCE_TEST_VAL)
        s.assertEquals(gb.z0, s.Z0_TEST_VAL)

        gb = GaussianBeam(
            s.AMPLITUDE_TEST_VAL, 
            s.WAVELNEGTH_TEST_VAL, 
            divergence=s.DIVERGENCE_TEST_VAL, 
            z_waist=s.Z_WAIST_TEST_VAL)
        s.assertEquals(gb.W0, s.W0_TEST_VAL)
        s.assertEquals(gb.z0, s.Z0_TEST_VAL)

        gb = GaussianBeam(
            s.AMPLITUDE_TEST_VAL, 
            s.WAVELNEGTH_TEST_VAL, 
            divergence=s.DIVERGENCE_TEST_VAL, 
            W0=s.W0_TEST_VAL)
        s.assertEquals(gb.z_waist, s.Z_WAIST_TEST_VAL)
        s.assertEquals(gb.z0, s.Z0_TEST_VAL)

        #TODO: test q parameter

