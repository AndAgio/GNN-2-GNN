import os
import logging


class Logger():
    def __init__(self, log_folder, name, log_file_name):
        self.log_folder = os.path.join(os.getcwd(), log_folder)
        # If log folder doesn't exist define it
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        # Setup logger
        log_file = os.path.join(self.log_folder, log_file_name)
        if os.path.exists(log_file):
            os.remove(log_file)
        # Setup the logger file
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log(self, message, level=logging.INFO):
        if level == logging.INFO:
            self.info(message)
        elif level == logging.DEBUG:
            self.debug(message)
        elif level == logging.ERROR:
            self.error(message)
        else:
            raise ValueError('Unable to handle {} logging level!'.format(level))

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)


if __name__ == '__main__':
    # first file logger
    logger = Logger('../log', 'first_logger', 'first_logfile.log')
    logger.info('This is just info message')

    # second file logger
    super_logger = Logger('../log', 'second_logger', 'second_logfile.log')
    super_logger.error('This is an error message')