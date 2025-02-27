import sys
from logging import (
    StreamHandler,
    Formatter,
    Logger,
    DEBUG
)

def set_logger(logger: Logger):
    logger.setLevel(DEBUG)

    # handler = StreamHandler(sys.stdout)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    
    formatter = Formatter('[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)