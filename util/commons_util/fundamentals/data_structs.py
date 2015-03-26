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
