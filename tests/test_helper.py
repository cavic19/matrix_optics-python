import unittest
from matrixopt.helper import *


class TestCountMatching(unittest.TestCase):
    def test_count_matching_success(self):
        test_list = [1, 2, 3, 5,1,8 ,11,0,1,-5,58,1232,0,0,0,1]
        expected = 5
        actual = count_matching(test_list, lambda x: x >= 5)
        self.assertEquals(expected, actual)

        test_list2 = ["a", "b", "c", "d", "e", "a", "b", "c", "a"]
        expected = 3
        actual = count_matching(test_list2, lambda x: x == "a")
        self.assertEquals(expected, actual)