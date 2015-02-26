import datetime

__author__ = 'Danyang'

class Timer(object):
    def __init__(self):
        self.s = None
        self.e = None

    def start(self):
        self.s = datetime.datetime.now()
        return self.s

    def end(self):
        self.e = datetime.datetime.now()
        return self.e - self.s