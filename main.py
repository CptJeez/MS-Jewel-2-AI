import traceback
import time
from logger import setup_logger
from window_manager import WindowManager
from jewel_detector import JewelDetector
from jewel_classifier import JewelClassifier
from jewel_ai import JewelAI

def main():
    """Main function to run the jewel detection and AI analysis"""
    # Set up logger
    logger = setup_logger()
    logger.info("Microsoft Jewel 2 Detection starting...")
    
    try:
        # Initialize window manager
        window_manager = WindowManager(logger)
        
        # Initialize jewel detector
        jewel_detector = JewelDetector(logger)
        
        # Initialize jewel classifier
        jewel_classifier = JewelClassifier(logger)
        
        # Initialize jewel AI (this will start the live display)
        jewel_ai = JewelAI(logger)
        
        try:
            # Position the game window
            success = window_manager.find_and_position_window()
            
            if success:
                logger.info("Window positioning completed successfully!")
                
                # Enter main loop for continuous analysis
                while True:
                    try:
                        # Capture the game board
                        logger.info("Capturing game board...")
                        img = window_manager.capture_game_board()
                        
                        if img is not None:
                            # Detect jewels
                            logger.info("Detecting jewels...")
                            jewel_grid = jewel_detector.detect_jewels(img)
                            
                            if jewel_grid is not None:
                                # Classify jewels by color
                                logger.info("Classifying jewels by color...")
                                jewel_grid = jewel_classifier.classify_jewels(jewel_grid)
                                
                                # Display jewel grid
                                jewel_classifier.display_jewel_grid(jewel_grid)
                                
                                # Analyze board and find possible moves
                                logger.info("Analyzing board for possible moves...")
                                possible_moves = jewel_ai.analyze_board(jewel_grid, img)
                                
                                logger.info(f"Found {len(possible_moves)} possible moves")
                                logger.info("Waiting for next capture cycle...")
                            else:
                                logger.error("Failed to detect jewels.")
                        else:
                            logger.error("Failed to capture game board.")
                        
                        # Wait before next capture (adjust as needed)
                        time.sleep(2.0)
                        
                    except KeyboardInterrupt:
                        logger.info("Keyboard interrupt detected. Exiting...")
                        break
                    except Exception as e:
                        logger.error(f"Error in capture cycle: {e}")
                        logger.error(traceback.format_exc())
                        time.sleep(5.0)  # Wait longer after an error
            else:
                logger.error("Failed to position window. Please check that Microsoft Jewel 2 is running.")
        
        finally:
            # Clean up resources
            jewel_ai.cleanup()
    
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    main()