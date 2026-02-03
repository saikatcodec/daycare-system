import logging
import os
from datetime import datetime

LOG_DIR = f'{datetime.now().strftime('%Y_%m_%d_%I_%M')}'
LOG_FILE = os.path.join('logs', LOG_DIR, LOG_DIR + '.log')

os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)

logging.basicConfig(
    format="[ %(%(asctime)s) ] - %(name)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    filename=LOG_FILE,
    level=logging.INFO
)