__author__ = 'Danyang'
import logging
import sys

class LoggerFactory(object):
    def getConsoleLogger(self, cls_name, level=logging.DEBUG):
        lgr = logging.getLogger(cls_name)
        lgr.setLevel(level)
        if not lgr.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            lgr.addHandler(ch)
        return lgr