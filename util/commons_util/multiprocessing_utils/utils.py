__author__ = 'Danyang'
from multiprocessing import Value
class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        with self.val.get_lock():
            return self.val.value
