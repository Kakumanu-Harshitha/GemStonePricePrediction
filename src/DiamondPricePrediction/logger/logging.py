import logging
import os
from datetime import datetime

# Create unique log filename based on current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Path where all logs will be stored
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)  # Create logs folder if it doesn't exist

# Full path for the log file
LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure logging format and file output
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILEPATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)
