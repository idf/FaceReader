from unittest import TestCase
from util.commons_util.fundamentals.data_structs import *
__author__ = 'Danyang'


class TestDataStructs(TestCase):
    def test_argsort(self):
        A = [3, 2, 1]
        ret = argsort(A)
        self.assertEqual(ret, [2, 1, 0])

    def test_excel_column(self):
        col = ExcelColumn()
        self.assertEqual(list(col.columns(800))[-1], 'adt')


class TestDisplayer(TestCase):
    def test_display(self):
        dis = Displayer()
        class A(object):
            def __init__(self):
                self.a = "abc"
                self.b = 1.0

        class B(object):
            def __init__(self):
                self.a = A()
                self.b = 1.0
        b = B()

        self.assertEqual(str(dis.dump(b)), '{"a": {"a": "abc", "b": 1.0}, "b": 1.0}')
        b_str = """{
    "a": {
        "a": "abc",
        "b": 1.0
    },
    "b": 1.0
}"""
        self.assertEqual(dis.display(b), b_str)
