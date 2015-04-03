from util.commons_util.logger_utils.timer import Timer

__author__ = 'Danyang'


def timestamp(func):
    """
    time the execution time of a function

    :param func: the function, whose result you would like to cached based on input arguments
    """
    def ret(*args):
        timer = Timer()
        timer.start()
        result = func(*args)
        print timer.end()
        return result
    return ret


def print_func_name(func):
    """
    print the current executing function name
    possible use:
    >>> print sys._getframe().f_code.co_name
    :param func: the function, whose result you would like to cached based on input arguments
    """
    def ret(*args):
        print func.func_name
        result = func(*args)
        return result
    return ret
