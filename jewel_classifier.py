import os
import traceback
import numpy as np
import cv2

class JewelClassifier:
    def __init__(self, logger):
        self.logger = logger
        self.output_dir = "jewel_detection"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define color ranges for jewel types (BGR format)
        self.color_ranges = {
            'red': {
                'lower': np.array([0, 0, 130]),
                'upper': np.array([120, 120, 255])
            },
            'blue': {
                'lower': np.array([120, 40, 0]),     # Expanded blue range
                'upper': np.array([255, 180, 120])
            },
            'green': {
                'lower': np.array([40, 130, 40]),
                'upper': np.array([180, 255, 180])
            },
            'yellow': {
                'lower': np.array([0, 130, 130]),
                'upper': np.array([120, 255, 255])
            },
            'purple': {
                'lower': np.array([120, 0, 120]),    # Expanded purple range
                'upper': np.array([255, 120, 255])
            }
        }
    
    def create_color_reference(self):
        """Create a color reference image for debugging"""
        self.logger.info("Creating color reference image...")
        
        try:
            # Create a canvas for color samples
            sample_img_height = 100
            sample_img_width = 100 * len(self.color_ranges)
            color_sample_img = np.zeros((sample_img_height, sample_img_width, 3), dtype=np.uint8)
            
            # Draw color samples for each jewel type
            x = 0
            for color_name, ranges in self.color_ranges.items():
                # Calculate center color
                center_color = ((ranges['lower'] + ranges['upper']) / 2).astype(int)
                
                # Draw color rectangle
                cv2.rectangle(color_sample_img, (x, 0), (x + 100, sample_img_height), 
                             (int(center_color[0]), int(center_color[1]), int(center_color[2])), -1)
                
                # Add color name
                cv2.putText(color_sample_img, color_name, (x + 10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                x += 100
            
            # Save color sample image
            cv2.imwrite(os.path.join(self.output_dir, "color_samples.png"), color_sample_img)
            return color_sample_img
        
        except Exception as e:
            self.logger.error(f"Error creating color reference: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def identify_jewel_type(self, color):
        """Identify jewel type based on its color"""
        try:
            best_match = None
            min_distance = float('inf')
            
            # First check if color is within any range
            for color_name, ranges in self.color_ranges.items():
                if np.all(color >= ranges['lower']) and np.all(color <= ranges['upper']):
                    # Calculate distance to center of range for tie-breaking
                    center = (ranges['lower'] + ranges['upper']) / 2
                    distance = np.sum((color - center) ** 2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = color_name
            
            # If no match found, use fallback method
            if best_match is None:
                # Find closest color range center
                for color_name, ranges in self.color_ranges.items():
                    center = (ranges['lower'] + ranges['upper']) / 2
                    distance = np.sum((color - center) ** 2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = color_name
            
            return best_match
        
        except Exception as e:
            self.logger.error(f"Error identifying jewel type: {e}")
            self.logger.error(traceback.format_exc())
            return "unknown"
    
    def create_color_grid_visualization(self, grid):
        """Create a visualization of the jewel grid with colors"""
        self.logger.info("Creating color grid visualization...")
        
        try:
            grid_size = 50
            grid_img = np.zeros((8 * grid_size, 8 * grid_size, 3), dtype=np.uint8)
            grid_img.fill(20)  # Dark background
            
            # Process each jewel in the grid
            for row in range(8):
                for col in range(8):
                    jewel = grid[row][col]
                    
                    if jewel is not None and 'color' in jewel:
                        try:
                            # Get jewel color
                            color = jewel['color']
                            
                            # Determine jewel type if not already assigned
                            if 'type' not in jewel:
                                jewel['type'] = self.identify_jewel_type(color)
                            
                            # Draw jewel color in grid visualization
                            x1 = col * grid_size + 5
                            y1 = row * grid_size + 5
                            x2 = (col + 1) * grid_size - 5
                            y2 = (row + 1) * grid_size - 5
                            
                            # Draw rectangle with actual color
                            cv2.rectangle(grid_img, (x1, y1), (x2, y2), 
                                         (int(color[0]), int(color[1]), int(color[2])), -1)
                            
                            # Add jewel type label with color indication (first letter)
                            text_color = (255, 255, 255)  # Default white text
                            
                            # For dark colors (like blue/purple), make text more visible
                            if jewel['type'] in ['blue', 'purple']:
                                text_color = (255, 255, 0)  # Yellow text for blue/purple
                                
                            cv2.putText(grid_img, jewel['type'][0].upper(), (x1 + 15, y1 + 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
                            
                        except Exception as e:
                            self.logger.error(f"Error drawing jewel at ({row},{col}): {e}")
                    else:
                        # Draw empty cell indicator
                        x1 = col * grid_size + 5
                        y1 = row * grid_size + 5
                        cv2.putText(grid_img, "?", (x1 + 20, y1 + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
            
            # Save color grid visualization
            cv2.imwrite(os.path.join(self.output_dir, "jewel_colors.png"), grid_img)
            return grid_img
        
        except Exception as e:
            self.logger.error(f"Error creating color grid visualization: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def classify_jewels(self, grid):
        """Classify all jewels in the grid by color"""
        self.logger.info("Starting jewel classification...")
        
        try:
            # Create color reference for debugging
            self.create_color_reference()
            
            # Classify each jewel in the grid
            for row in range(8):
                for col in range(8):
                    jewel = grid[row][col]
                    if jewel is not None and 'color' in jewel:
                        jewel['type'] = self.identify_jewel_type(jewel['color'])
            
            # Create visualization of classified jewels
            self.create_color_grid_visualization(grid)
            
            self.logger.info("Jewel classification completed")
            return grid
        
        except Exception as e:
            self.logger.error(f"Error in jewel classification: {e}")
            self.logger.error(traceback.format_exc())
            return grid  # Return original grid even if classification fails
    
    def display_jewel_grid(self, grid):
        """Display the classified jewel grid in the console"""
        try:
            self.logger.info("Displaying jewel grid in console...")
            
            # First, display the jewel types
            self.logger.info("Detected Jewel Grid (Types):")
            for row in range(8):
                row_str = []
                for col in range(8):
                    jewel = grid[row][col]
                    if jewel is not None and 'type' in jewel:
                        row_str.append(f"{jewel['type']:8}")
                    else:
                        row_str.append("none    ")
                self.logger.info(" ".join(row_str))
            
            # Then, display color values for debugging
            self.logger.info("\nDetected Jewel Colors (BGR):")
            for row in range(8):
                row_str = []
                for col in range(8):
                    jewel = grid[row][col]
                    if jewel is not None and 'color' in jewel:
                        color = jewel['color']
                        row_str.append(f"({row},{col}):{int(color[0])},{int(color[1])},{int(color[2])}")
                    else:
                        row_str.append(f"({row},{col}):none")
                self.logger.info(" | ".join(row_str))
        
        except Exception as e:
            self.logger.error(f"Error displaying jewel grid: {e}")
            self.logger.error(traceback.format_exc())