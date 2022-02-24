import unittest
from matrixopt.beams import *


class TestGaussianBeam(unittest.TestCase):
    AMPLITUDE_TEST_VAL = 1
    WAIST_RADIUS_TEST_VAL = 1.2e-3 #1.2mm
    RAYLEIGH_RANGE_TEST_VAL = 11170.10721e-3 #11170.10721mm
    DIVERGENCE_TEST_VAL = 0.10743e-3 #0.10743mrad
    Z_WAIST_TEST_VAL = 0

    REFR_INDEX_TEST_VAL = 1.2
    WAVELNEGTH_TEST_VAL = 405e-9 #405nm

    def test_init_no_beam_param_fail(s):
        with s.assertRaises(ValueError):
            GaussianBeam(wave_length=s.WAVELNEGTH_TEST_VAL)

    def test_init_too_much_beam_params_fail(s):
        with s.assertRaises(ValueError):
            GaussianBeam(wave_length=s.WAVELNEGTH_TEST_VAL, zr=1, div=1)

    def test_init_divergence_presented_success(s):
        gb = GaussianBeam(wave_length=s.WAVELNEGTH_TEST_VAL, div=s.DIVERGENCE_TEST_VAL)
        s.assertAlmostEqual(gb.divergence, s.DIVERGENCE_TEST_VAL, 3)
        s.assertAlmostEqual(gb.waist_radius, s.WAIST_RADIUS_TEST_VAL, 3)
        s.assertAlmostEqual(gb.rayleigh_range, s.RAYLEIGH_RANGE_TEST_VAL, 3)


    def test_init_waist_radius_presented_success(s):
        gb = GaussianBeam(wave_length=s.WAVELNEGTH_TEST_VAL, w0=s.WAIST_RADIUS_TEST_VAL)
        s.assertAlmostEqual(gb.divergence, s.DIVERGENCE_TEST_VAL, 3)
        s.assertAlmostEqual(gb.waist_radius, s.WAIST_RADIUS_TEST_VAL, 3)
        s.assertAlmostEqual(gb.rayleigh_range, s.RAYLEIGH_RANGE_TEST_VAL, 3)

    def test_init_zr_presented_success(s):
        gb = GaussianBeam(wave_length=s.WAVELNEGTH_TEST_VAL, zr=s.RAYLEIGH_RANGE_TEST_VAL)
        s.assertAlmostEqual(gb.divergence, s.DIVERGENCE_TEST_VAL, 3)
        s.assertAlmostEqual(gb.waist_radius, s.WAIST_RADIUS_TEST_VAL, 3)
        s.assertAlmostEqual(gb.rayleigh_range, s.RAYLEIGH_RANGE_TEST_VAL, 3)
