import time
import pygetwindow as gw
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def find_and_position_jewel_window():
    """Find Microsoft Jewel 2 window, resize it, and position it at top-left corner"""
    
    # Define the window dimensions from the screenshot
    TARGET_WIDTH = 1437
    TARGET_HEIGHT = 832
    
    logging.info("Searching for Microsoft Jewel 2 window...")
    
    # Try multiple possible window titles since it might vary
    possible_titles = [
        "Solitaire & Casual Games"
    ]
    
    jewel_window = None
    
    # Try to find the window by iterating through possible titles
    for title in possible_titles:
        try:
            matching_windows = [win for win in gw.getAllTitles() if title in win]
            if matching_windows:
                window_title = matching_windows[0]
                jewel_window = gw.getWindowsWithTitle(window_title)[0]
                logging.info(f"Found window with title: {window_title}")
                break
        except Exception as e:
            logging.error(f"Error searching for '{title}': {e}")
    
    if not jewel_window:
        logging.error("Could not find Microsoft Jewel 2 window. "
                      "Please make sure the game is running.")
        return False
    
    # Activate the window and bring it to the foreground
    try:
        jewel_window.activate()
        time.sleep(0.5)  # Give it time to come to the foreground
        
        # Get current window state
        logging.info(f"Current window position: ({jewel_window.left}, {jewel_window.top})")
        logging.info(f"Current window size: {jewel_window.width}x{jewel_window.height}")
        
        # If the window is maximized, restore it first
        if jewel_window.isMaximized:
            logging.info("Window is maximized. Restoring to normal size...")
            jewel_window.restore()
            time.sleep(0.5)
        
        # Resize the window
        logging.info(f"Resizing window to {TARGET_WIDTH}x{TARGET_HEIGHT}...")
        jewel_window.resizeTo(TARGET_WIDTH, TARGET_HEIGHT)
        time.sleep(0.5)
        
        # Move the window to the top-left corner
        logging.info("Moving window to top-left corner (0, 0)...")
        jewel_window.moveTo(0, 0)
        
        logging.info("Window positioning complete!")
        logging.info(f"New window position: ({jewel_window.left}, {jewel_window.top})")
        logging.info(f"New window size: {jewel_window.width}x{jewel_window.height}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error manipulating window: {e}")
        return False

if __name__ == "__main__":
    logging.info("Microsoft Jewel 2 Window Manager starting...")
    
    # Make sure required packages are available
    try:
        import pygetwindow
    except ImportError:
        logging.error("Required package 'pygetwindow' is not installed.")
        logging.info("Please install it with: pip install pygetwindow")
        sys.exit(1)
    
    # Run the main function
    success = find_and_position_jewel_window()
    
    if success:
        logging.info("Window positioning completed successfully!")
    else:
        logging.error("Failed to position window. Please check that Microsoft Jewel 2 is running.")