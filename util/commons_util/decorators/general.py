from util.commons_util.logger_utils.Timer import Timer

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