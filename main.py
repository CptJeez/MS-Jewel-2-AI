import traceback
from logger import setup_logger
from window_manager import WindowManager
from jewel_detector import JewelDetector
from jewel_classifier import JewelClassifier

def main():
    """Main function to run the jewel detection"""
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
        
        # Position the game window
        success = window_manager.find_and_position_window()
        
        if success:
            logger.info("Window positioning completed successfully!")
            
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
                    
                    logger.info("Jewel detection completed! Check the jewel_detection folder for output images.")
                    return jewel_grid
                else:
                    logger.error("Failed to detect jewels.")
            else:
                logger.error("Failed to capture game board.")
        else:
            logger.error("Failed to position window. Please check that Microsoft Jewel 2 is running.")
        
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()