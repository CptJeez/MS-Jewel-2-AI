import cv2
import numpy as np
import threading
import time
import traceback
import pygetwindow as gw
import os

class LiveDisplay:
    def __init__(self, logger):
        self.logger = logger
        self.window_name = "Microsoft Jewel 2 AI"
        self.is_running = False
        self.display_thread = None
        self.current_image = None
        self.lock = threading.Lock()
        self.output_dir = "jewel_detection"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get screen dimensions
        screen_width = 1920  # Default value
        screen_height = 1080  # Default value
        try:
            # Try to get actual screen dimensions
            windows = gw.getAllWindows()
            if windows:
                # Get the first window's monitor size
                screen_width = gw._pygetwindow_win.GetSystemMetrics(0)
                screen_height = gw._pygetwindow_win.GetSystemMetrics(1)
                self.logger.info(f"Detected screen dimensions: {screen_width}x{screen_height}")
        except Exception as e:
            self.logger.warning(f"Could not detect screen dimensions, using defaults: {e}")
        
        # Set window dimensions
        self.window_width = 500
        self.window_height = 600
        
        # Calculate position (top-right corner)
        self.window_x = screen_width - self.window_width
        self.window_y = 0
    
    def start(self):
        """Start the display thread"""
        if not self.is_running:
            self.is_running = True
            self.display_thread = threading.Thread(target=self._display_loop)
            self.display_thread.daemon = True
            self.display_thread.start()
            self.logger.info("Live display started")
    
    def stop(self):
        """Stop the display thread"""
        self.is_running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
            self.display_thread = None
        cv2.destroyWindow(self.window_name)
        self.logger.info("Live display stopped")
    
    def update_image(self, img):
        """Update the displayed image"""
        with self.lock:
            # Resize the image to fit the window
            if img is not None:
                # Save original aspect ratio
                h, w = img.shape[:2]
                aspect_ratio = w / h
                
                # Calculate new dimensions to fit the window
                if aspect_ratio > (self.window_width / self.window_height):
                    # Image is wider
                    new_w = self.window_width
                    new_h = int(new_w / aspect_ratio)
                else:
                    # Image is taller
                    new_h = self.window_height
                    new_w = int(new_h * aspect_ratio)
                
                # Resize the image
                self.current_image = cv2.resize(img, (new_w, new_h))
                
                # Debugging
                debug_path = os.path.join(self.output_dir, "live_display_debug.png")
                cv2.imwrite(debug_path, self.current_image)
    
    def _display_loop(self):
        """Main display loop"""
        try:
            # Create the window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
            cv2.moveWindow(self.window_name, self.window_x, self.window_y)
            
            # Start with a blank image
            blank_image = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            cv2.putText(blank_image, "Waiting for game board...", (50, self.window_height//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            with self.lock:
                self.current_image = blank_image
            
            # Main display loop
            while self.is_running:
                with self.lock:
                    display_img = self.current_image.copy() if self.current_image is not None else blank_image
                
                cv2.imshow(self.window_name, display_img)
                
                # Position the window in case it was moved
                cv2.moveWindow(self.window_name, self.window_x, self.window_y)
                
                # Process window events and check for keypress
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC key
                    self.is_running = False
            
            cv2.destroyWindow(self.window_name)
            
        except Exception as e:
            self.logger.error(f"Error in display loop: {e}")
            self.logger.error(traceback.format_exc())
            self.is_running = False
    
    def create_moves_visualization(self, grid, moves, img):
        """Create a visualization of the moves and update the display"""
        try:
            # Create a copy of the original image
            moves_img = img.copy()
            
            # Draw arrows for each move
            for i, move in enumerate(moves):
                from_row, from_col = move['from_row'], move['from_col']
                to_row, to_col = move['to_row'], move['to_col']
                
                # Get the centers of the jewels
                if grid[from_row][from_col] is None or grid[to_row][to_col] is None:
                    continue
                    
                from_center = grid[from_row][from_col]['center']
                to_center = grid[to_row][to_col]['center']
                
                # Draw an arrow
                cv2.arrowedLine(
                    moves_img, 
                    from_center, 
                    to_center, 
                    (0, 0, 255),  # Red color
                    3,  # Thickness
                    tipLength=0.3  # Relative size of arrow tip
                )
                
                # Add text label with move number and score if available
                mid_x = (from_center[0] + to_center[0]) // 2
                mid_y = (from_center[1] + to_center[1]) // 2
                
                label = f"{i+1}"
                if 'score' in move:
                    label += f":{move['score']}"
                    
                cv2.putText(
                    moves_img,
                    label,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),  # White text
                    2
                )
            
            # Add information overlay at the top
            if moves:
                best_move = moves[0]
                info_text = f"Best Move: ({best_move['from_row']},{best_move['from_col']}) -> ({best_move['to_row']},{best_move['to_col']})"
                if 'score' in best_move:
                    info_text += f" Score: {best_move['score']}"
                
                # Add text at the top of the image
                cv2.rectangle(moves_img, (0, 0), (moves_img.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(
                    moves_img,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),  # White text
                    2
                )
            
            # Save the image for debugging
            cv2.imwrite(os.path.join(self.output_dir, "possible_moves.png"), moves_img)
            
            # Update the display
            self.update_image(moves_img)
            
            return moves_img
            
        except Exception as e:
            self.logger.error(f"Error creating moves visualization: {e}")
            self.logger.error(traceback.format_exc())
            return img