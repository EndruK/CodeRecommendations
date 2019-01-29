import logging


def init_logging(level, file):
    """
    initialize the logger for the complete run-through
    :param level: logging level (info, debug, warn, critical)
    :param file: file-path to store the log to
    :return:
    """

    if level == "info":
        log_level = logging.INFO
    elif level == "debug":
        log_level = logging.DEBUG
    elif level == "warn":
        log_level = logging.WARN
    elif level == "critical":
        log_level = logging.CRITICAL
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(file),
            logging.StreamHandler()
        ]
    )
    logging.debug("initialized logger")
    logging.info("Log Level = " + level)
