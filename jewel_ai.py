import os
import cv2
import numpy as np
import traceback
from live_display import LiveDisplay

class JewelAI:
    def __init__(self, logger):
        self.logger = logger
        self.output_dir = "jewel_detection"
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the live display
        self.display = LiveDisplay(logger)
        # Start the display
        self.display.start()
    
    def find_valid_moves(self, grid, img):
        """Find all valid moves in the grid (swaps that create matches)"""
        self.logger.info("Finding valid moves...")
        valid_moves = []
        
        try:
            # Loop through the grid
            for row in range(8):
                for col in range(8):
                    # Check each of the four possible directions
                    directions = [
                        ('right', 0, 1),   # right
                        ('down', 1, 0),    # down
                        ('left', 0, -1),   # left
                        ('up', -1, 0)      # up
                    ]
                    
                    for direction, dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        
                        # Skip if out of bounds
                        if not (0 <= new_row < 8 and 0 <= new_col < 8):
                            continue
                        
                        # Try the swap
                        if self._is_valid_move(grid, row, col, new_row, new_col):
                            valid_moves.append({
                                'from_row': row,
                                'from_col': col,
                                'to_row': new_row,
                                'to_col': new_col,
                                'direction': direction
                            })
            
            self.logger.info(f"Found {len(valid_moves)} valid moves")
            return valid_moves
            
        except Exception as e:
            self.logger.error(f"Error finding valid moves: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def _is_valid_move(self, grid, row1, col1, row2, col2):
        """Check if swapping jewels at (row1, col1) and (row2, col2) creates a match"""
        try:
            # Skip if either jewel is None
            if grid[row1][col1] is None or grid[row2][col2] is None:
                return False
            
            # Get jewel types
            type1 = grid[row1][col1].get('type', '')
            type2 = grid[row2][col2].get('type', '')
            
            # Skip if types are the same (no need to swap)
            if type1 == type2:
                return False
            
            # Create a copy of the grid with the swap
            grid_copy = self._create_grid_copy(grid)
            
            # Perform the swap (we only need to swap the types for checking)
            grid_copy[row1][col1]['type'], grid_copy[row2][col2]['type'] = grid_copy[row2][col2]['type'], grid_copy[row1][col1]['type']
            
            # Check if the swap creates a match
            if self._check_for_matches(grid_copy, row1, col1) or self._check_for_matches(grid_copy, row2, col2):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking move validity: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _create_grid_copy(self, grid):
        """Create a copy of the grid for move simulation"""
        grid_copy = [[None for _ in range(8)] for _ in range(8)]
        
        for row in range(8):
            for col in range(8):
                if grid[row][col] is not None:
                    # Create a shallow copy of the jewel dict - we only need the 'type' for matching
                    grid_copy[row][col] = grid[row][col].copy()
        
        return grid_copy
    
    def _check_for_matches(self, grid, row, col):
        """Check if there's a match of 3 or more at position (row, col)"""
        try:
            if grid[row][col] is None:
                return False
            
            jewel_type = grid[row][col].get('type', '')
            
            # Check horizontal match
            count = 1
            # Check left
            c = col - 1
            while c >= 0 and grid[row][c] is not None and grid[row][c].get('type', '') == jewel_type:
                count += 1
                c -= 1
            
            # Check right
            c = col + 1
            while c < 8 and grid[row][c] is not None and grid[row][c].get('type', '') == jewel_type:
                count += 1
                c += 1
            
            if count >= 3:
                return True
            
            # Check vertical match
            count = 1
            # Check up
            r = row - 1
            while r >= 0 and grid[r][col] is not None and grid[r][col].get('type', '') == jewel_type:
                count += 1
                r -= 1
            
            # Check down
            r = row + 1
            while r < 8 and grid[r][col] is not None and grid[r][col].get('type', '') == jewel_type:
                count += 1
                r += 1
            
            return count >= 3
            
        except Exception as e:
            self.logger.error(f"Error checking for matches: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def rank_moves(self, grid, valid_moves):
        """Rank moves by potential score (more matches = higher score)"""
        self.logger.info("Ranking moves by potential score...")
        
        try:
            scored_moves = []
            
            for move in valid_moves:
                from_row, from_col = move['from_row'], move['from_col']
                to_row, to_col = move['to_row'], move['to_col']
                
                # Create a grid copy
                grid_copy = self._create_grid_copy(grid)
                
                # Perform the swap
                grid_copy[from_row][from_col]['type'], grid_copy[to_row][to_col]['type'] = grid_copy[to_row][to_col]['type'], grid_copy[from_row][from_col]['type']
                
                # Calculate the score for this move
                score = self._calculate_move_score(grid_copy, from_row, from_col, to_row, to_col)
                
                # Add score to the move
                move_with_score = move.copy()
                move_with_score['score'] = score
                scored_moves.append(move_with_score)
            
            # Sort moves by score (descending)
            scored_moves.sort(key=lambda m: m['score'], reverse=True)
            
            return scored_moves
            
        except Exception as e:
            self.logger.error(f"Error ranking moves: {e}")
            self.logger.error(traceback.format_exc())
            return valid_moves
    
    def _calculate_move_score(self, grid, row1, col1, row2, col2):
        """Calculate the score for a move based on match length"""
        try:
            score = 0
            
            # Check for matches at both swapped positions
            for row, col in [(row1, col1), (row2, col2)]:
                if grid[row][col] is None:
                    continue
                
                jewel_type = grid[row][col].get('type', '')
                
                # Check horizontal match length
                h_count = 1
                # Check left
                c = col - 1
                while c >= 0 and grid[row][c] is not None and grid[row][c].get('type', '') == jewel_type:
                    h_count += 1
                    c -= 1
                
                # Check right
                c = col + 1
                while c < 8 and grid[row][c] is not None and grid[row][c].get('type', '') == jewel_type:
                    h_count += 1
                    c += 1
                
                # Add to score if there's a match
                if h_count >= 3:
                    # Score is match length squared (e.g., 3=9, 4=16, 5=25)
                    score += h_count * h_count
                
                # Check vertical match length
                v_count = 1
                # Check up
                r = row - 1
                while r >= 0 and grid[r][col] is not None and grid[r][col].get('type', '') == jewel_type:
                    v_count += 1
                    r -= 1
                
                # Check down
                r = row + 1
                while r < 8 and grid[r][col] is not None and grid[r][col].get('type', '') == jewel_type:
                    v_count += 1
                    r += 1
                
                # Add to score if there's a match
                if v_count >= 3:
                    score += v_count * v_count
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating move score: {e}")
            self.logger.error(traceback.format_exc())
            return 0
    
    def analyze_board(self, grid, img):
        """Main method to analyze the board and find possible moves"""
        self.logger.info("Starting board analysis...")
        
        try:
            # Find valid moves
            valid_moves = self.find_valid_moves(grid, img)
            
            # Rank the moves by potential score
            ranked_moves = self.rank_moves(grid, valid_moves)
            
            # Display the moves in the live window
            self.display.create_moves_visualization(grid, ranked_moves, img)
            
            # Log the top moves
            if ranked_moves:
                self.logger.info(f"Top move: From ({ranked_moves[0]['from_row']},{ranked_moves[0]['from_col']}) "
                                f"to ({ranked_moves[0]['to_row']},{ranked_moves[0]['to_col']}) "
                                f"Score: {ranked_moves[0].get('score', 0)}")
                
                # Log a few more top moves if available
                for i in range(1, min(5, len(ranked_moves))):
                    self.logger.info(f"Move {i+1}: From ({ranked_moves[i]['from_row']},{ranked_moves[i]['from_col']}) "
                                    f"to ({ranked_moves[i]['to_row']},{ranked_moves[i]['to_col']}) "
                                    f"Score: {ranked_moves[i].get('score', 0)}")
            
            return ranked_moves
            
        except Exception as e:
            self.logger.error(f"Error in board analysis: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def cleanup(self):
        """Cleanup resources when done"""
        # Stop the live display
        self.display.stop()