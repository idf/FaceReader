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


def memoize_force(func):
    """
    Similar to memoize, but force the hash by using its string value
    But caching performance may be a issue

    :param func: the function, whose result you would like to cached based on input arguments
    """
    cache = {}
    def ret(*args):
        k = str(args)
        if k not in cache:
            cache[k] = func(*args)
        return cache[k]
    return ret


def memoize_iterable(func):
    """
    Similar to memoize, but force the hash by using its tuple value
    The arguments for the function must be iterable
    """
    cache = {}
    def ret(*args):
        k = tuple(args)
        if k not in cache:
            cache[k] = func(*args)
        return cache[k]
    return ret