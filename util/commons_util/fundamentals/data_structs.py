__author__ = 'Danyang'


def argsort(A, f=None):
    """
    :type A: list
    :param A:
    :return:
    """
    n = len(A)
    if f == None:
        f = lambda k: A[k]
    return sorted(range(n), key=f)


class ExcelColumn(object):
    def __init__(self):
        self.cur = None
        self.restore()

    def restore(self):
        self.cur = []

    def _plus(self, cur, idx):
        if idx>=len(cur):
            cur.append('a')
        elif cur[idx]<'z':
            cur[idx] = chr(ord(cur[idx])+1)
        else:
            cur[idx] = 'a'
            self._plus(cur, idx+1)

    def columns(self, n):
        """
        generator for excel columns
        :param n: decimal count for excel col
        :return:
        """
        self.restore()
        for i in xrange(n):
            self._plus(self.cur, 0)
            yield ''.join(reversed(self.cur))