__author__ = 'Danyang'
import os
import re
class FileUtils(object):
    @classmethod
    def rename(cls):
        """
        template
        :return:
        """
        find = r".+(?P<time>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.png$)"
        dir = os.path.dirname(os.path.realpath(__file__))
        for r, dirs, fs in os.walk(dir):
            for f in fs:
                m = re.search(find, f)
                if m:
                    f1 = m.group("time")
                    os.rename(f, f1)
                    print "%s -> %s" % (f, f1)

class CmdUtils(object):
    @classmethod
    def execute(cls, cmd):
        os.system(cmd)