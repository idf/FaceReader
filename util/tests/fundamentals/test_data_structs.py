from unittest import TestCase
from util.commons_util.fundamentals.data_structs import *
__author__ = 'Danyang'


class TestDataStructs(TestCase):
    def test_argsort(self):
        A = [3, 2, 1]
        ret = argsort(A)
        self.assertEqual(ret, [2, 1, 0])