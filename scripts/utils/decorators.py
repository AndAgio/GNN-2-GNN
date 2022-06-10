import time
import os
import logging


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print('{} took {:.3f} s'.format(method.__name__, te - ts))
        return result

    return timed


def timeit_and_log(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger_folder = os.path.join('log', 'timeit_decorator_logger')
        logger = define_logger(logger_folder, 'timeit_decorator_logger', 'timeit_decorator_logger.log')
        logger.info('{} took {:.3f} s'.format(method.__name__.upper(), te - ts))
        return result

    return timed


def define_logger(log_folder, log_file_name, name):
    log_folder = os.path.join(os.getcwd(), log_folder)
    # If log folder doesn't exist define it
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # Setup logger
    log_file = os.path.join(log_folder, log_file_name)
    # Setup the logger file
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
