#!/usr/bin/env python
import logging
import logging.handlers
import os.path


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create a file handler
    logging_file = name + ".log"
    # handler = logging.handlers.TimedRotatingFileHandler(logging_file,
    #                                                     when="m",
    #                                                     interval=1,
    #                                                     backupCount=5)
    should_roll_over = os.path.isfile(logging_file)
    handler = logging.handlers.RotatingFileHandler(
        logging_file, mode='a', backupCount=5)
    if should_roll_over:  # log already exists, roll over!
        handler.doRollover()

    logger.addHandler(handler)
    # handler = logging.FileHandler(logging_file)
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the file handler to the logger
    logger.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    return logger
