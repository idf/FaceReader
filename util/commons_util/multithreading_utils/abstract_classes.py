__author__ = 'Danyang'
import threading
from decorators.classes import *


def print_msg(name, msg):
    print "%s says: %s" % (name, msg)

class AbstractThread(threading.Thread):
    @Override(threading.Thread)
    def __init__(self, name, production=False):
        super(AbstractThread, self).__init__()
        self.name = name
        self.production = production

    def print_msg(self, msg):
        print_msg(self.name, msg)