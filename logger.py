import logging
import os

def setup_logger():
    """Set up and configure the logger"""
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join("logs", "jewel_detector.log")),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()