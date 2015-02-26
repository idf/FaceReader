__author__ = 'Danyang'
def memoize(func):
    """
    the function must not modify or rely on external state 
    the function should be stateless. 
    usage: @memoize as function annotation

    :param func: the function, whose result you would like to cached based on input arguments
    """
    cache = {}
    def ret(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return ret