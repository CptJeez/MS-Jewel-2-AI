import os
import traceback
import numpy as np
import cv2

class JewelDetector:
    def __init__(self, logger):
        self.logger = logger
        self.output_dir = "jewel_detection"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def preprocess_image(self, img):
        """Preprocess the image to enhance jewel visibility"""
        self.logger.info("Preprocessing image...")
        
        try:
            # Save original image
            cv2.imwrite(os.path.join(self.output_dir, "original.png"), img)
            
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Enhance brightness and contrast
            enhanced = cv2.convertScaleAbs(img, alpha=1.3, beta=15)
            
            # Normalize the image to improve consistency across the board
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Save enhanced image
            cv2.imwrite(os.path.join(self.output_dir, "enhanced.png"), enhanced)
            
            return enhanced, hsv
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            self.logger.error(traceback.format_exc())
            return img, None
    
    def create_masks(self, enhanced, hsv):
        """Create binary and color masks for jewel detection"""
        self.logger.info("Creating detection masks...")
        
        try:
            # Create morphological kernel - smaller kernel for finer details
            kernel = np.ones((3, 3), np.uint8)
            larger_kernel = np.ones((5, 5), np.uint8)
            
            # METHOD 1: Adaptive thresholding for better handling of lighting variations
            self.logger.info("Applying adaptive thresholding...")
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            # Use adaptive thresholding to handle variations across the board
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            # Invert because adaptive threshold gives white background and black objects
            binary = cv2.bitwise_not(binary)
            
            # Apply morphological operations
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, larger_kernel)
            
            # Save binary mask
            cv2.imwrite(os.path.join(self.output_dir, "binary_mask.png"), binary)
            
            # METHOD 2: Multi-channel color detection with expanded ranges
            self.logger.info("Applying multi-channel color detection...")
            
            # Create an empty mask for the color detection result
            color_mask = np.zeros_like(gray)
            
            # Define HSV ranges for all jewel colors with wider ranges
            color_ranges = [
                # Blue jewels (wider hue range)
                {'name': 'blue', 'lower': (85, 40, 40), 'upper': (135, 255, 255)},
                # Purple jewels
                {'name': 'purple', 'lower': (135, 30, 30), 'upper': (180, 255, 255)},
                # Red jewels (wraps around hue 0)
                {'name': 'red1', 'lower': (0, 40, 40), 'upper': (10, 255, 255)},
                {'name': 'red2', 'lower': (170, 40, 40), 'upper': (180, 255, 255)},
                # Green jewels
                {'name': 'green', 'lower': (35, 40, 40), 'upper': (85, 255, 255)},
                # Yellow jewels
                {'name': 'yellow', 'lower': (15, 40, 40), 'upper': (35, 255, 255)}
            ]
            
            # Process each color range and combine masks
            for color_range in color_ranges:
                lower = np.array(color_range['lower'])
                upper = np.array(color_range['upper'])
                mask = cv2.inRange(hsv, lower, upper)
                
                # Save individual color masks for debugging
                color_debug_path = os.path.join(self.output_dir, f"mask_{color_range['name']}.png")
                cv2.imwrite(color_debug_path, mask)
                
                # Combine with the overall color mask
                color_mask = cv2.bitwise_or(color_mask, mask)
            
            # Apply morphological operations to clean up the color mask
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, larger_kernel)
            
            # Save the color mask for debugging
            cv2.imwrite(os.path.join(self.output_dir, "color_mask.png"), color_mask)
            
            # METHOD 3: Edge detection for finding jewel boundaries
            self.logger.info("Applying edge detection...")
            
            # Apply Canny edge detection
            edges = cv2.Canny(enhanced, 50, 150)
            
            # Dilate edges to connect broken contours
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Save edge mask
            cv2.imwrite(os.path.join(self.output_dir, "edge_mask.png"), edges)
            
            # Combine all masks for robust detection
            combined_mask = cv2.bitwise_or(binary, color_mask)
            combined_mask = cv2.bitwise_or(combined_mask, edges)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, larger_kernel)
            
            # Save combined mask
            cv2.imwrite(os.path.join(self.output_dir, "combined_mask.png"), combined_mask)
            
            return combined_mask
        except Exception as e:
            self.logger.error(f"Error creating masks: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def find_contours(self, mask, img):
        """Find and filter contours from the mask"""
        self.logger.info("Finding contours...")
        
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.logger.info(f"Found {len(contours)} contours from combined mask")
            
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Calculate expected jewel size
            grid_width = width / 8
            grid_height = height / 8
            expected_area = (grid_width * grid_height) * 0.5  # 50% of cell area is a reasonable target
            
            # Filter contours by size with more flexible thresholds
            min_contour_area = expected_area * 0.1  # 10% of expected area
            max_contour_area = expected_area * 2.0  # 200% of expected area
            
            self.logger.info(f"Expected jewel area: {expected_area:.1f} pixels")
            self.logger.info(f"Area filter range: {min_contour_area:.1f} to {max_contour_area:.1f} pixels")
            
            valid_contours = []
            for c in contours:
                area = cv2.contourArea(c)
                if min_contour_area < area < max_contour_area:
                    valid_contours.append(c)
            
            self.logger.info(f"After filtering: {len(valid_contours)} valid contours")
            
            # If we have too few contours, use a grid-based approach instead
            if len(valid_contours) < 40:  # Less than 60% of expected jewels
                self.logger.warning(f"Too few contours detected ({len(valid_contours)}). Using grid-based approach.")
                return self.generate_grid_based_contours(img)
            
            # Draw all contours for debugging
            contour_image = img.copy()
            cv2.drawContours(contour_image, valid_contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(self.output_dir, "contours.png"), contour_image)
            
            return valid_contours
        except Exception as e:
            self.logger.error(f"Error finding contours: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def generate_grid_based_contours(self, img):
        """Generate contours based on the expected grid layout"""
        self.logger.info("Generating grid-based contours...")
        
        try:
            height, width = img.shape[:2]
            grid_width = width / 8
            grid_height = height / 8
            
            synthetic_contours = []
            
            # Create contours for each cell in the grid
            for row in range(8):
                for col in range(8):
                    # Calculate cell center
                    center_x = int((col + 0.5) * grid_width)
                    center_y = int((row + 0.5) * grid_height)
                    
                    # Calculate cell boundaries (slightly smaller than actual cell)
                    radius = int(min(grid_width, grid_height) * 0.4)
                    
                    # Create a circular contour
                    points = []
                    for angle in range(0, 360, 10):
                        x = center_x + int(radius * np.cos(np.radians(angle)))
                        y = center_y + int(radius * np.sin(np.radians(angle)))
                        points.append([x, y])
                    
                    contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
                    synthetic_contours.append(contour)
            
            # Draw synthetic contours for debugging
            contour_image = img.copy()
            cv2.drawContours(contour_image, synthetic_contours, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(self.output_dir, "synthetic_contours.png"), contour_image)
            
            self.logger.info(f"Generated {len(synthetic_contours)} synthetic contours")
            return synthetic_contours
        except Exception as e:
            self.logger.error(f"Error generating grid-based contours: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def create_grid_visualization(self, img):
        """Create a grid visualization for debugging"""
        try:
            height, width = img.shape[:2]
            grid_width = width / 8
            grid_height = height / 8
            
            # Create a blank image for the grid visualization
            grid_image = np.zeros_like(img)
            for i in range(9):
                # Draw horizontal lines
                y = int(i * grid_height)
                cv2.line(grid_image, (0, y), (width, y), (0, 255, 0), 1)
                # Draw vertical lines
                x = int(i * grid_width)
                cv2.line(grid_image, (x, 0), (x, height), (0, 255, 0), 1)
            
            # Overlay grid on original image
            grid_overlay = cv2.addWeighted(img, 0.7, grid_image, 0.3, 0)
            cv2.imwrite(os.path.join(self.output_dir, "grid_overlay.png"), grid_overlay)
            
            return grid_width, grid_height
        except Exception as e:
            self.logger.error(f"Error creating grid visualization: {e}")
            self.logger.error(traceback.format_exc())
            height, width = img.shape[:2]
            return width / 8, height / 8
    
    def extract_jewel_info(self, contours, img, grid_width, grid_height):
        """Extract jewel information from contours"""
        self.logger.info("Extracting jewel information from contours...")
        jewels = []
        img_with_centers = img.copy()
        
        for i, contour in enumerate(contours):
            try:
                # Get the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # Fallback to bounding box center if moments method fails
                    x, y, w, h = cv2.boundingRect(contour)
                    cX = x + w // 2
                    cY = y + h // 2
                
                # Calculate grid position (row, col)
                col = int(cX / grid_width)
                row = int(cY / grid_height)
                
                # Ensure row and col are within valid range (0-7)
                col = max(0, min(7, col))
                row = max(0, min(7, row))
                
                # Sample color around the center
                region_size = 15  # Size of sampling region
                y_start = max(0, cY - region_size)
                y_end = min(img.shape[0], cY + region_size)
                x_start = max(0, cX - region_size)
                x_end = min(img.shape[1], cX + region_size)
                
                region = img[y_start:y_end, x_start:x_end]
                
                # Skip if region is empty
                if region.size == 0:
                    continue
                    
                # Reshape region to a list of pixels
                pixels = region.reshape(-1, 3)
                
                # Filter out very dark pixels (background)
                non_dark_pixels = pixels[np.sum(pixels, axis=1) > 120]
                
                if len(non_dark_pixels) > 0:
                    # Calculate dominant color (median is more robust than mean)
                    color = np.median(non_dark_pixels, axis=0).astype(int)
                else:
                    # Fallback if no non-dark pixels found
                    color = np.median(pixels, axis=0).astype(int)
                
                # Store jewel info
                jewels.append({
                    'id': i,
                    'row': row,
                    'col': col,
                    'center': (cX, cY),
                    'contour': contour,
                    'color': color
                })
                
                # Draw the center point and grid position on the image
                cv2.circle(img_with_centers, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(img_with_centers, f"{row},{col}", (cX + 10, cY),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                self.logger.error(f"Error processing contour {i}: {e}")
                self.logger.error(traceback.format_exc())
        
        # Save the annotated image
        cv2.imwrite(os.path.join(self.output_dir, "jewel_centers.png"), img_with_centers)
        
        return jewels
    
    def organize_jewels_into_grid(self, jewels, grid_width, grid_height):
        """Organize jewels into an 8x8 grid"""
        self.logger.info("Organizing jewels into grid...")
        grid = [[None for _ in range(8)] for _ in range(8)]
        
        # First pass: place jewels in the grid
        for jewel in jewels:
            row, col = jewel['row'], jewel['col']
            
            # If we already have a jewel at this position, keep the one closest to the grid center
            if grid[row][col] is not None:
                existing_jewel = grid[row][col]
                existing_center = existing_jewel['center']
                new_center = jewel['center']
                
                # Calculate ideal grid center
                ideal_x = (col + 0.5) * grid_width
                ideal_y = (row + 0.5) * grid_height
                
                # Calculate distances to ideal center
                existing_dist = (existing_center[0] - ideal_x) ** 2 + (existing_center[1] - ideal_y) ** 2
                new_dist = (new_center[0] - ideal_x) ** 2 + (new_center[1] - ideal_y) ** 2
                
                # Keep the one closer to the ideal center
                if new_dist < existing_dist:
                    grid[row][col] = jewel
            else:
                grid[row][col] = jewel
        
        return grid
    
    def fill_missing_positions(self, grid, img, grid_width, grid_height):
        """Fill in missing positions in the grid"""
        # Identify missing positions
        missing_positions = []
        for row in range(8):
            for col in range(8):
                if grid[row][col] is None:
                    missing_positions.append((row, col))
        
        if missing_positions:
            self.logger.warning(f"Missing jewels at positions: {missing_positions}")
            
            # Try to fill in missing positions using a grid-based approach
            self.logger.info("Attempting to fill in missing positions...")
            
            # For each missing position, sample color at the cell center
            for row, col in missing_positions:
                try:
                    # Calculate center of the cell
                    center_x = int((col + 0.5) * grid_width)
                    center_y = int((row + 0.5) * grid_height)
                    
                    # Use a larger sampling radius for missing jewels
                    region_size = 25  # Increased from 15 to 25 for more robust sampling
                    
                    # Sample in multiple locations around the center
                    sample_points = [
                        (center_x, center_y),                     # Center
                        (center_x - 10, center_y),                # Left
                        (center_x + 10, center_y),                # Right
                        (center_x, center_y - 10),                # Top
                        (center_x, center_y + 10),                # Bottom
                        (center_x - 7, center_y - 7),             # Top-left
                        (center_x + 7, center_y - 7),             # Top-right
                        (center_x - 7, center_y + 7),             # Bottom-left
                        (center_x + 7, center_y + 7),             # Bottom-right
                    ]
                    
                    # Collect color samples from all points
                    all_samples = []
                    
                    for px, py in sample_points:
                        # Define sampling region
                        y_start = max(0, py - region_size)
                        y_end = min(img.shape[0], py + region_size)
                        x_start = max(0, px - region_size)
                        x_end = min(img.shape[1], px + region_size)
                        
                        region = img[y_start:y_end, x_start:x_end]
                        
                        if region.size > 0:
                            # Reshape and filter dark pixels
                            pixels = region.reshape(-1, 3)
                            non_dark_pixels = pixels[np.sum(pixels, axis=1) > 100]
                            
                            if len(non_dark_pixels) > 0:
                                # Add these pixels to our collection
                                all_samples.extend(non_dark_pixels)
                    
                    # If we have samples, create a synthetic jewel entry
                    if all_samples:
                        samples_array = np.array(all_samples)
                        color = np.median(samples_array, axis=0).astype(int)
                        
                        # Create a synthetic jewel
                        synthetic_jewel = {
                            'id': 1000 + len(missing_positions),
                            'row': row,
                            'col': col,
                            'center': (center_x, center_y),
                            'color': color,
                            'synthetic': True  # Mark as synthetically detected
                        }
                        
                        # Add to grid
                        grid[row][col] = synthetic_jewel
                        
                        self.logger.info(f"Filled in missing jewel at ({row},{col}) with synthetic detection")
                except Exception as e:
                    self.logger.error(f"Error filling in missing jewel at ({row},{col}): {e}")
                    self.logger.error(traceback.format_exc())
        
        return grid
    
    def special_handling_for_first_column(self, grid, img, grid_width, grid_height):
        """Special handling for the first column which is often problematic"""
        self.logger.info("Applying special handling for first column...")
        
        for row in range(8):
            try:
                if grid[row][0] is None or 'synthetic' in grid[row][0]:
                    self.logger.info(f"Special handling for first column at row {row}")
                    
                    # Sample at fixed positions known to work well for first column
                    center_x = int(grid_width * 0.5)
                    center_y = int((row + 0.5) * grid_height)
                    
                    # Enhanced sampling with larger region
                    region_size = 30
                    y_start = max(0, center_y - region_size)
                    y_end = min(img.shape[0], center_y + region_size)
                    x_start = max(0, center_x - region_size)
                    x_end = min(img.shape[1], center_x + region_size)
                    
                    region = img[y_start:y_end, x_start:x_end]
                    
                    if region.size > 0:
                        # Simple color averaging instead of K-means
                        pixels = region.reshape(-1, 3)
                        pixels = pixels[np.sum(pixels, axis=1) > 100]
                        
                        if len(pixels) > 0:
                            # Use median color
                            color = np.median(pixels, axis=0).astype(int)
                            
                            # Create or update jewel
                            if grid[row][0] is None:
                                grid[row][0] = {
                                    'id': 2000 + row,
                                    'row': row,
                                    'col': 0,
                                    'center': (center_x, center_y),
                                    'color': color,
                                    'special': True  # Mark as special handling
                                }
                            else:
                                grid[row][0]['color'] = color
                                grid[row][0]['special'] = True
                            
                            self.logger.info(f"Special detection for first column at ({row},0)")
            except Exception as e:
                self.logger.error(f"Error in special handling for first column at row {row}: {e}")
                self.logger.error(traceback.format_exc())
        
        return grid
    
    def create_final_visualization(self, grid, img):
        """Create final visualization with circles around jewels"""
        self.logger.info("Creating final visualization...")
        refined_img = img.copy()
        
        # Process each cell in the grid
        for row in range(8):
            for col in range(8):
                jewel = grid[row][col]
                
                if jewel is not None:
                    try:
                        # Draw circle at the center of the jewel
                        cX, cY = jewel['center']
                        
                        # Determine circle color based on detection method
                        circle_color = (0, 255, 0)  # Default green
                        if jewel.get('synthetic', False):
                            circle_color = (0, 0, 255)  # Red for synthetic detections
                        elif jewel.get('special', False):
                            circle_color = (255, 0, 255)  # Magenta for special handling
                        
                        # Draw a circle
                        cv2.circle(refined_img, (cX, cY), 35, circle_color, 2)
                        
                        # Add row/col text
                        cv2.putText(refined_img, f"{row},{col}", (cX - 20, cY + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        self.logger.error(f"Error drawing jewel at ({row},{col}): {e}")
        
        # Save the refined image
        cv2.imwrite(os.path.join(self.output_dir, "jewel_detection.png"), refined_img)
        
        return refined_img
    
    def detect_jewels(self, img):
        """Main jewel detection function"""
        self.logger.info("Starting jewel detection process...")
        
        try:
            # Step 1: Preprocess the image
            enhanced, hsv = self.preprocess_image(img)
            
            # Step 2: Create masks for jewel detection
            combined_mask = self.create_masks(enhanced, hsv)
            if combined_mask is None:
                self.logger.error("Failed to create masks")
                return None
            
            # Step 3: Find and filter contours
            valid_contours = self.find_contours(combined_mask, img)
            
            # Step 4: Create grid visualization for debugging
            grid_width, grid_height = self.create_grid_visualization(img)
            
            # Step 5: Extract jewel information from contours
            jewels = self.extract_jewel_info(valid_contours, img, grid_width, grid_height)
            
            # Step 6: Organize jewels into a grid
            grid = self.organize_jewels_into_grid(jewels, grid_width, grid_height)
            
            # Step 7: Fill in missing positions
            grid = self.fill_missing_positions(grid, img, grid_width, grid_height)
            
            # Step 8: Special handling for first column
            grid = self.special_handling_for_first_column(grid, img, grid_width, grid_height)
            
            # Step 9: Create final visualization
            self.create_final_visualization(grid, img)
            
            self.logger.info("Jewel detection completed successfully")
            return grid
        
        except Exception as e:
            self.logger.error(f"Error in jewel detection: {e}")
            self.logger.error(traceback.format_exc())
            return None