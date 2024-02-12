import logging
import os
from datetime import datetime 
import time

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'logs',LOG_FILE)

# Creates logs directory and logfile if not present else it appends the information
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Whenever the logging.info() is intiated and adds the logs to the log file
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s %(message)s",
    level= logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging Started")
    
