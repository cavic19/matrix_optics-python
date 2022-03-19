import unittest
from optix.beams import *
import numpy as np

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


    def test_beam_radius_ndarray(self):
        zr = 1
        w0 =  np.sqrt((zr * 1) / (np.pi * 1))
        gb = GaussianBeam(1, zr=zr)
        array = np.linspace(0, 10, 100)
        expected = [w0*np.sqrt(1 + (z/zr)**2) for z in array]

        actual = list(gb.beam_radius(array))

        self.assertListEqual(actual, expected)

    def test_curviture_ndarray(self):
        zr = 1
        gb = GaussianBeam(1, zr=zr)
        array = np.linspace(0.1, 10, 100)
        expected = [z*(1 + (zr/z)**2) for z in array]

        actual = list(gb.curviture(array))

        self.assertListEqual(actual, expected)

    def test_cbeam_param_ndarray(self):
        zr = 1
        gb = GaussianBeam(1, zr=zr)
        array = np.linspace(0.1, 10, 100)
        expected = [complex(z, zr) for z in array]

        actual = list(gb.cbeam_parameter(array))

        self.assertListEqual(actual, expected)

    def test_str(self):
        zr = 1
        w0 = np.sqrt(zr*1/np.pi)
        div = 1 / (np.pi * w0)
        gb = GaussianBeam(1, zr=zr)

        expected = f"""
        amplitude\t=\t1,
        wavelength\t=\t1000000000 nm,
        waist_loc\t=\t0 cm,
        waist_rad\t=\t{w0*10**3} mm,
        rayleigh_r\t=\t{zr*10**3} mm,
        divergence\t=\t{div*10**3} mrad
        """
        actual = str(gb)

        self.assertMultiLineEqual(actual, expected)
