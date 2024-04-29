import numpy as np
from collections import deque

'''
Helper Function that determines that max possible distance value for a given matrix
@param matrix_width: {int} no. of columns in the matrix
@param matrix_height: {int} no. of rows in the matrix
@returns max distance value for any cell
'''
def get_max_distance(matrix_width: int, matrix_height: int):
    return max(matrix_width, matrix_height) * 2 + 1

'''
@param start_cells: list of { row: <row_index>, col: <col_index> } of starting positions
@param matrix_width: {int} no. of columns in the matrix
@param matrix_height: {int} no. of rows in the matrix
@param obstacles: (Optional) list of tuples (row, col) for blockages/inaccessible cells
@param stop_coords: (Optional) list of tuples (row, col) when the positions are all filled with values, can terminate the algorithm early
@param normalize: (Optional){Boolean} True if normalisation to [0,1] range should be applied
@returns matrix that has been filled with manhattan distance to all foods
'''
def flash_flood(start_cells: list, matrix_width: int, matrix_height: int, stop_coords=[], obstacles=[], normalize=False):
    ## HELPER FUNCTIONS
    '''
    Checks if a matrix cell has been processed and assigned a distance value
    @param row: row index of cell
    @param col: column index of cell
    @returns {Boolean} True if processed, False if not
    '''
    def is_visited(row, col):
        return DISTANCE_MATRIX[row][col] < MAX_VALUE
    
    '''
    Checks if the cell position is blocked or can be assigned a distance value
    @param row: row index of cell
    @param col: column index of cell
    @returns {Boolean} True if is obstacle, False if not
    '''
    def is_obstacle(row, col):
        return (row, col) in obstacles
    
    '''
    Gets the 4 valid neighbours at the 4 cardinals (Up/Down/Left/Right) of cell
    (Valid: Not out of bounds, Not obstacle)
    @param row: row index of cell
    @param col: column index of cell
    @returns visited: {List} of visited neighbour cell indexes as { row: <row_index>, col: <col_index> }
    @returns unvisited: {List} of unvisited neighbour cell indexes as { row: <row_index>, col: <col_index> }
    '''
    def get_neighbours(row: int, col: int):
        visited = []
        unvisited = []
        # Above
        above_row = row - 1
        if above_row >= 0 and not is_obstacle(above_row, col):
            top_cell = (above_row, col)
            if is_visited(above_row, col):      visited.append(top_cell)
            elif top_cell not in UNVISITED:     unvisited.append(top_cell)
        # Below
        below_row = row + 1
        if below_row <= matrix_height-1 and not is_obstacle(below_row, col):
            bottom_cell = (below_row, col)
            if is_visited(below_row, col):      visited.append(bottom_cell)
            elif bottom_cell not in UNVISITED:  unvisited.append(bottom_cell)
        # Left
        left_col = col - 1
        if left_col >= 0 and not is_obstacle(row, left_col):
            left_cell = (row, left_col)
            if is_visited(row, left_col):       visited.append(left_cell)
            elif left_cell not in UNVISITED:    unvisited.append(left_cell)
        # Right
        right_col = col + 1
        if right_col <= matrix_width-1 and not is_obstacle(row, right_col):
            right_cell = (row, right_col)
            if is_visited(row, right_col):      visited.append(right_cell)
            elif right_cell not in UNVISITED:   unvisited.append(right_cell)       
        return visited, unvisited
    
    # Ensure it is greater than the max distance possible on the board
    MAX_VALUE = get_max_distance(matrix_width, matrix_height)

    
    # Place starting coordinates into queue
    UNVISITED = deque(start_cells)
    stop_coords = stop_coords.copy()

    # Create empty matrix to flood, initialise to max number
    DISTANCE_MATRIX = np.full((matrix_height, matrix_width), MAX_VALUE)

    # Main Loop: keep looping until queue is empty
    counter = 1
    while len(stop_coords) > 0 and bool(UNVISITED):
        row, col = UNVISITED.popleft()
        
        # # Get neighbours with values at the 4 cardinals        
        visited_neighbours, unvisited_neighbours = get_neighbours(row, col)

        # Derive cell value
        if (row, col) in start_cells:   # For initial food positions
            DISTANCE_MATRIX[row][col] = 0
        else:
            neighbour_values = [ DISTANCE_MATRIX[row][col] for (row, col) in visited_neighbours ]
            DISTANCE_MATRIX[row][col] = min(neighbour_values) + 1
        
        if (row, col) in stop_coords: stop_coords.remove((row, col))

        # Place unvisited neighbours into the queue
        UNVISITED.extend(unvisited_neighbours)
        counter += 1

    # Normalise
    if normalize:
        DISTANCE_MATRIX = np.divide(DISTANCE_MATRIX, MAX_VALUE)
    return DISTANCE_MATRIX

'''
For all other parameters, refer to flash_flood
@param pac_position: pacman's (row, col) coordinates
@returns list of (row, col) tuples of matrix cells with the closest distance to the start_cells
'''
def get_closest_cardinals(pacman_position: tuple, start_cells: list, matrix_width, matrix_height, obstacles=[], normalize=False):
    # Get Pacman's 4 cardinals
    pacman_row, pacman_col = pacman_position
    north_row, north_col = (max(pacman_row-1, 0), pacman_col)
    south_row, south_col = (min(pacman_row+1, matrix_height-1), pacman_col)
    west_row, west_col = (pacman_row, max(pacman_col-1, 0))
    east_row, east_col = (pacman_row, min(pacman_col+1, matrix_width-1))
    pacman_cardinals = [(north_row, north_col), (south_row, south_col), (west_row, west_col), (east_row, east_col)]
    
    distances = flash_flood(start_cells, matrix_width, matrix_height, 
                stop_coords=pacman_cardinals, 
                obstacles=obstacles, normalize=normalize)

    # Get Pacman Cardinal Values
    cardinal_distances = [
        distances[north_row][north_col],
        distances[south_row][south_col],
        distances[west_row][west_col],
        distances[east_row][east_col]
    ]
    min_distance = min(cardinal_distances)
    MAX_VALUE = get_max_distance(matrix_width, matrix_height)
    cardinals_with_min_dist = []

    # Only collect the cardinal directions with the minimum distance
    if min_distance < MAX_VALUE:
        for row, col in pacman_cardinals:
            if distances[row][col] == min_distance:
                cardinals_with_min_dist.append((row, col))

    return cardinals_with_min_dist
