import logging


def init_logging(level, file):
    """
    initialize the logger for the complete run-through
    :param level: logging level (info, debug, warn, critical)
    :param file: file-path to store the log to
    :return:
    """

    if level == "info":
        level = logging.INFO
    elif level == "debug":
        level = logging.DEBUG
    elif level == "warn":
        level = logging.WARN
    elif level == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(file),
            logging.StreamHandler()
        ]
    )
    logging.debug("initialized logger")
    logging.info("Log Level = " + level)
