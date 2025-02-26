import time
import traceback
import numpy as np
import cv2
from PIL import ImageGrab
import pygetwindow as gw

class WindowManager:
    def __init__(self, logger):
        self.logger = logger
        # Game board coordinates after window positioning
        self.BOARD_TOP_LEFT = (825, 304)
        self.BOARD_BOTTOM_RIGHT = (1660, 1167)
        # Target window dimensions
        self.TARGET_WIDTH = 1437
        self.TARGET_HEIGHT = 832

    def find_and_position_window(self):
        """Find Microsoft Jewel 2 window, resize it, and position it at top-left corner"""
        self.logger.info("Starting window detection...")
        
        # Try multiple possible window titles since it might vary
        possible_titles = [
            "Solitaire & Casual Games"
        ]
        
        jewel_window = None
        
        # List all window titles for debugging
        all_titles = gw.getAllTitles()
        self.logger.info(f"All window titles: {all_titles}")
        
        # Try to find the window by iterating through possible titles
        for title in possible_titles:
            try:
                self.logger.info(f"Looking for windows with title containing: '{title}'")
                matching_windows = [win for win in all_titles if title in win]
                self.logger.info(f"Matching windows: {matching_windows}")
                
                if matching_windows:
                    window_title = matching_windows[0]
                    jewel_window = gw.getWindowsWithTitle(window_title)[0]
                    self.logger.info(f"Found window with title: {window_title}")
                    break
            except Exception as e:
                self.logger.error(f"Error searching for '{title}': {e}")
                self.logger.error(traceback.format_exc())
        
        if not jewel_window:
            self.logger.error("Could not find Microsoft Jewel 2 window. "
                          "Please make sure the game is running.")
            return False
        
        # Activate the window and bring it to the foreground
        try:
            self.logger.info("Activating window...")
            jewel_window.activate()
            time.sleep(0.5)  # Give it time to come to the foreground
            
            # Get current window state
            self.logger.info(f"Current window position: ({jewel_window.left}, {jewel_window.top})")
            self.logger.info(f"Current window size: {jewel_window.width}x{jewel_window.height}")
            
            # If the window is maximized, restore it first
            if jewel_window.isMaximized:
                self.logger.info("Window is maximized. Restoring to normal size...")
                jewel_window.restore()
                time.sleep(0.5)
            
            # Resize the window
            self.logger.info(f"Resizing window to {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}...")
            jewel_window.resizeTo(self.TARGET_WIDTH, self.TARGET_HEIGHT)
            time.sleep(0.5)
            
            # Move the window to the top-left corner
            self.logger.info("Moving window to top-left corner (0, 0)...")
            jewel_window.moveTo(0, 0)
            
            self.logger.info("Window positioning complete!")
            self.logger.info(f"New window position: ({jewel_window.left}, {jewel_window.top})")
            self.logger.info(f"New window size: {jewel_window.width}x{jewel_window.height}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error manipulating window: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def capture_game_board(self):
        """Capture the game board area and return it as an OpenCV image"""
        self.logger.info("Starting screen capture...")
        
        try:
            # Give the window time to fully render after positioning
            time.sleep(1)
            
            self.logger.info(f"Capturing screen region: {self.BOARD_TOP_LEFT} to {self.BOARD_BOTTOM_RIGHT}")
            # Capture the screen region
            screenshot = ImageGrab.grab(bbox=(
                self.BOARD_TOP_LEFT[0], 
                self.BOARD_TOP_LEFT[1], 
                self.BOARD_BOTTOM_RIGHT[0], 
                self.BOARD_BOTTOM_RIGHT[1]
            ))
            
            self.logger.info("Converting screenshot to OpenCV format...")
            # Convert PIL image to OpenCV format (RGB to BGR)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Calculate dimensions
            board_width = self.BOARD_BOTTOM_RIGHT[0] - self.BOARD_TOP_LEFT[0]
            board_height = self.BOARD_BOTTOM_RIGHT[1] - self.BOARD_TOP_LEFT[1]
            self.logger.info(f"Captured game board: {board_width}x{board_height} pixels")
            
            return img
        
        except Exception as e:
            self.logger.error(f"Error capturing game board: {e}")
            self.logger.error(traceback.format_exc())
            return None