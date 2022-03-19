from optix.beams.experiment_processing import *
import unittest
import numpy as np

class TestExtractM2(unittest.TestCase):
    def test_gaussian_beam_input(self):
        matrix = []
        for y in range(500):
            for x in range(500):
                matrix[y][x] = 0
        assert False, "Not implemented"