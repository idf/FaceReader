__author__ = 'Danyang'
import logging
import sys
class LoggerFactory(object):
    def getConsoleLogger(self, cls_name):
        lgr = logging.getLogger(cls_name)
        lgr.setLevel(logging.CRITICAL)
        if not lgr.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            lgr.addHandler(ch)
        return lgr
