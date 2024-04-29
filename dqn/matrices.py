import numpy as np
from dqn.dist_utils import get_closest_cardinals

'''
Function that prepares all the matrices for feeding into the DQN
@param state: Takes in the GameState object for the current state
@param NUM_CHANNELS: Number of channels
@param VISIBLE_LIMIT: number of squares pacman can see in the 4 cardinal directions (N/S/E/W)
@param AUDIBLE_LIMIT: number of squares pacman can sense around him (food and ghosts only)
@param CHANNEL_RADIUS: length of 1 cardinal
@param channel_height: number of rows in the channel
@param channel_width: number of columns in the channel
@param is_partially_observable: (Optional) Set to True if there is partial observability. Defaults to False
@param is_sliced: (Optional) Set to True if matrix size is to be sliced to CHANNEL_RADIUS. Defaults to False
@returns the observation matrix with NUM_CHANNEL amount of channels
'''
def prepare_observation_matrix(state, NUM_CHANNELS: int, channel_height: int, channel_width: int,
                               VISIBLE_LIMIT: int, AUDIBLE_LIMIT: int, CHANNEL_RADIUS: int,
                               is_partially_observable=False, is_sliced=False):
    layout_height = state.data.layout.height
    layout_width = state.data.layout.width
    '''
    Creates a numpy array of size channel_height x channel_width
    initialised to zeros (for fully observable) or UNOBSERVABLE_CELL (for partially observable)
    @param mask: (Optional) {Boolean} True if all invisible range should be set to -1 instead of 0
    '''
    def initialise_channel_matrix(mask=False):
        if mask:
            return np.full((layout_height, layout_width), UNOBSERVABLE_CELL, dtype=float)
        else:
            return np.zeros((layout_height, layout_width), dtype=float)
    '''
    Converts Grid coordinates into matrix row and column indexes
    @param x: x-coordinate of Grid
    @param y: y-coordinate of Grid
    @returns (row, col) of matrix
    '''
    def convert_coords_to_matrix_indexes(x, y):
        '''
        Take Note: Grid Coordinates are transposed from actual layout appearance (rotated clockwise by 90 degrees)
        (I.e. each row in the Grid object refers to each column in the grid layout)
        '''
        return layout_height - 1 - y, x # inverse y-axis (layout grid coordinates start from bottom left as (0,0))

    # Constants for filling matrices
    UNOBSERVABLE_CELL = 0      # out of range as well as obstructed by walls
    OBSERVABLE_EMPTY = 0.5     # can observe but nothing in cell
    OBSERVABLE_EXIST = 1       # found entity in cell
    EXISTS = 1      # (for state grids) the entity is in the cell
    EMPTY = 0       # (for state grids) the entity is not in the cell

    # Fully-Observable Environment Information
    PACMAN_POSITION = state.getPacmanPosition()
    BOOL_WALLS_GRID = state.data.layout.walls       # 2D array same size as the layout, True/False values
    BOOL_FOOD_GRID = state.data.food         # 2D array same size as the layout, True/False values
    CAPSULE_COORDS = state.data.capsules     # List of (x, y) coordinates as tuples

    # Partially-Observable Environment Information
    VISIBLE_COORDS = []  # To contain the absolute (x, y) coordinates of visible cells
    AUDIBLE_COORDS = []  # To contain the absolute (x, y) coordinates of audible cells
    WALL_INDEXES = set()    # List of (row, col) indexes of wall positions to be used for the distance matrix
    FOOD_INDEXES = set()    # List of (row, col) indexes of food positions to be used for the distance matrix
    GHOST_INDEXES = set()   # List of (row, col) indexes of ghost positions to be used for the distance matrix

    # Channel Matrices to be fed to CNN (initialise all cells to 0)
    PACMAN_CHANNEL = initialise_channel_matrix(mask=is_partially_observable)
    WALLS_CHANNEL = initialise_channel_matrix(mask=is_partially_observable)
    FOOD_CHANNEL = initialise_channel_matrix(mask=is_partially_observable)
    GHOSTS_CHANNEL = initialise_channel_matrix(mask=is_partially_observable)
    SCARED_GHOSTS_CHANNEL = initialise_channel_matrix(mask=is_partially_observable)
    CAPSULES_CHANNEL = initialise_channel_matrix(mask=is_partially_observable)
    GHOSTS_DIST_CHANNEL = initialise_channel_matrix(mask=is_partially_observable)


    ############################## PARTIALLY OBSERVABLE ###############################
    '''
    Creates a list of (x, y) coordinates observable by pacman.
    Also gets the row and column ranges of the bounding box wrapping the entire
    observable zone (for slicing)
    @returns: dictionary containing:
        - min_x: leftmost column index visible by pacman
        - max_x: rightmost column index visible by pacman
        - min_y: topmost row index visible by pacman
        - max_y: bottommost row index visible by pacman
    '''
    def derive_observable_coordinates():
        bounding_radius = CHANNEL_RADIUS
        x, y = PACMAN_POSITION
        # Outer Bounding Box, including cardinals
        outer_min_x = max(x-bounding_radius, 0)
        outer_max_x = min(x+bounding_radius, layout_width-1)
        outer_min_y = max(y-bounding_radius, 0)
        outer_max_y = min(y+bounding_radius, layout_height-1)
        bounding_box = { 'min_x': outer_min_x, 'max_x': outer_max_x, 'min_y': outer_min_y, 'max_y': outer_max_y }

        # Inner Bounding Box (audible radius only)
        bounding_radius = AUDIBLE_LIMIT
        inner_min_x = max(x-bounding_radius, 0)
        inner_max_x = min(x+bounding_radius, layout_width-1)
        inner_min_y = max(y-bounding_radius, 0)
        inner_max_y = min(y+bounding_radius, layout_height-1)

        # Collect audible coordinates
        for x in range(inner_min_x, inner_max_x + 1):
            for y in range(inner_min_y, inner_max_y + 1):
                AUDIBLE_COORDS.append((x, y))
        pacman_x = int(PACMAN_POSITION[0])
        pacman_y = int(PACMAN_POSITION[1])

        # Indicate Visible Coordinates until you hit a wall
        '''
        Helper function to collect (x, y) coordinate tuples visible by pacman
        (Excludes cells already blocked by a wall)
        @param start: starting for loop index
        @param end: starting for loop index
        @param direction: north/south/east/west (only for printout purpose)
        @param fix_axis: 'x' or 'y' depending on which axis is to remain constant
        @param loop_backwards: {Boolean} True if the for loop is to decrement instead of increment
        '''
        def indicate_visible_coords_for_direction(start, end, direction, fix_axis='y', loop_backwards=False):
            for i in range(start, end, -1 if loop_backwards else 1):
                if fix_axis == 'x':   x, y = (pacman_x, i)
                else:                 x, y = (i, pacman_y)
                VISIBLE_COORDS.append((x, y))
                if BOOL_WALLS_GRID[x][y] == EXISTS:
                    break

        # Pacman's Position
        VISIBLE_COORDS.append(PACMAN_POSITION)
        # WEST
        indicate_visible_coords_for_direction(
            pacman_x - 1, outer_min_x - 1, 'west', fix_axis='y', loop_backwards=True)
        # EAST
        indicate_visible_coords_for_direction(
            pacman_x + 1, outer_max_x + 1, 'east', fix_axis='y', loop_backwards=False)
        # SOUTH
        indicate_visible_coords_for_direction(
            pacman_y - 1, outer_min_y - 1, 'south', fix_axis='x', loop_backwards=True)
        # NORTH
        indicate_visible_coords_for_direction(
            pacman_y + 1, outer_max_y + 1, 'north', fix_axis='x', loop_backwards=False)

        return bounding_box

    '''
    Initialises the observable cells of all channel matrices to OBSERVABLE_EMPTY
    (Precondition: Must call derive_observable_coordinates() at least once before)
    '''
    def initialise_observable_zones():
        for x, y in VISIBLE_COORDS:
            row, col = convert_coords_to_matrix_indexes(x, y)
            PACMAN_CHANNEL[row][col] = OBSERVABLE_EMPTY
            WALLS_CHANNEL[row][col] = OBSERVABLE_EMPTY
            FOOD_CHANNEL[row][col] = OBSERVABLE_EMPTY
            GHOSTS_CHANNEL[row][col] = OBSERVABLE_EMPTY
            SCARED_GHOSTS_CHANNEL[row][col] = OBSERVABLE_EMPTY
            CAPSULES_CHANNEL[row][col] = OBSERVABLE_EMPTY
            GHOSTS_DIST_CHANNEL[row][col] = OBSERVABLE_EMPTY

    '''
    Slices the layout sized channel into a smaller channel size
    @returns the sliced matrix
    '''
    def slice_channel(channel_matrix, bounding_box=None, pad_value=UNOBSERVABLE_CELL):
        if bounding_box == None:
            return channel_matrix   # Nothing to do, return
        # Slice to bounding box size
        min_x = bounding_box['min_x']
        max_x = bounding_box['max_x']
        '''
        Since grid coordinates are the inverse of the matrix coordinates, vertically,
        swap min and max and convert them to positive (x, y) coordinates.
        (Don't use convert_coords_to_matrix_indexes(). It gives negative relative indexes)
        '''
        min_y = layout_height - (bounding_box['max_y'] + 1)
        max_y = layout_height - (bounding_box['min_y'] + 1)

        '''
        Expands the slice area on the side of the minimum or maximum bound depending on
        which side of the board the observable window is leaning on
        @param min_bound: index of the minimum bound on said axis
        @param min_bound: index of the maximum bound on said axis
        @param channel_length: length of the side of the channel being targeted
        @param layout_length: length of the side of the layout being targeted
        @returns {tuple} (adjusted min bound, adjusted max bound)
        '''
        def adjust_bounds_to_channel_length(min_bound, max_bound, channel_length=0, layout_length=0):
            diff = max_bound - min_bound + 1
            pad_info = None
            # Pad Slice to Channel Size where required
            if diff < channel_length:
                padding_required = channel_length - diff
                if max_bound == (layout_length-1):
                    # Observable Window is leaning on max side, Pad min side
                    min_bound = max(0, min_bound - padding_required)
                    pad_info = { 'side': 'min' }
                elif min_bound == 0:
                    # Observable Window is leaning on min side, Pad max side
                    max_bound = min(channel_length-1, max_bound + padding_required)
                    pad_info = { 'side': 'max' }
            return min_bound, max_bound, pad_info

        # Re-adjust min and max bounds in order to generate a matrix of the desired channel size
        min_x, max_x, pad_x = adjust_bounds_to_channel_length(min_x, max_x, channel_length=channel_width, layout_length=layout_width)
        min_y, max_y, pad_y = adjust_bounds_to_channel_length(min_y, max_y, channel_length=channel_height, layout_length=layout_height)
        sliced = channel_matrix[min_y:max_y+1,min_x:max_x+1]

        '''
        If still not desired channel size (usually because the layout itself is too small),
        pad with numpy
        '''
        if not sliced.shape == (channel_height, channel_width):
            padding_required_x = channel_width - sliced.shape[1]
            padding_required_y = channel_height - sliced.shape[0]
            if pad_x:
                if pad_x['side'] == 'min':
                    sliced = np.pad(sliced, [(0, 0), (padding_required_x, 0)], mode='constant', constant_values=pad_value)
                else:
                    sliced = np.pad(sliced, [(0, 0), (0, padding_required_x)], mode='constant', constant_values=pad_value)
            if pad_y:
                if pad_y['side'] == 'min':
                    sliced = np.pad(sliced, [(padding_required_y, 0), (0, 0)], mode='constant', constant_values=pad_value)
                else:
                    sliced = np.pad(sliced, [(0, padding_required_y), (0, 0)], mode='constant', constant_values=pad_value)
        return sliced

    '''
    Sets positions for Ghosts/Scared Ghosts to OBSERVABLE_EXIST in their relevant matrices
    (Only sets for ghosts that are visible or audible)
    @param bounded: (Optional) {Boolean} True when pacman has limited vision. Defaults to False
    '''
    def set_ghost_matrices(bounded=False):
        for agentState in state.data.agentStates:
            pos = agentState.configuration.getPosition()
            matrix_row, col = convert_coords_to_matrix_indexes(int(pos[0]), int(pos[1]))

            # Ignore ghosts outside of detection range
            agent_is_visible = pos in VISIBLE_COORDS
            agent_is_audible = pos in AUDIBLE_COORDS
            if not bounded or (bounded and (agent_is_visible or agent_is_audible)):
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        SCARED_GHOSTS_CHANNEL[matrix_row][col] = OBSERVABLE_EXIST
                    else:
                        GHOSTS_CHANNEL[matrix_row][col] = OBSERVABLE_EXIST
            elif not agentState.isPacman:
                GHOST_INDEXES.add((matrix_row, col))

    '''
    Collects the row and column indexes of food, wall and ghost positions
    Sets them directly into WALL_INDEXES, FOOD_INDEXES
    '''
    def get_distance_indexes():
        for x in range(layout_width):
            for y in range(layout_height):
                row, col = convert_coords_to_matrix_indexes(x, y)
                if BOOL_FOOD_GRID[x][y]:    FOOD_INDEXES.add((row, col))
                if BOOL_WALLS_GRID[x][y]:   WALL_INDEXES.add((row, col))

    '''
    Sets positions for Food/Walls/Capsules to OBSERVABLE_EXIST in their relevant matrices
    @param x: x-coordinate of Grid
    @param y: y-coordinate of Grid
    '''
    def set_matrix_values(x, y):
        row, col = convert_coords_to_matrix_indexes(x, y)
        # Set cell values to 1 if there is such an entity in that grid cell
        if BOOL_WALLS_GRID[x][y]:
            WALLS_CHANNEL[row][col] = OBSERVABLE_EXIST
        else:
            WALLS_CHANNEL[row][col] = OBSERVABLE_EMPTY
        if BOOL_FOOD_GRID[x][y]:
            FOOD_CHANNEL[row][col] = OBSERVABLE_EXIST
        else:
            FOOD_CHANNEL[row][col] = OBSERVABLE_EMPTY
        CAPSULES_CHANNEL[row][col] = OBSERVABLE_EXIST if (x, y) in CAPSULE_COORDS else OBSERVABLE_EMPTY

    '''
    Calls dist_utils to generate a distance matrix to pacman then marks the directions nearest to
    far away food with pseudofood (i.e. as OBSERVABLE_EXIST)
    '''
    def set_nearest_food_directions():
            get_distance_indexes()     # For food
            pacman_location = convert_coords_to_matrix_indexes(PACMAN_POSITION[0], PACMAN_POSITION[1])
            nearest_food_directions = get_closest_cardinals(pacman_location,
                            list(FOOD_INDEXES), layout_width, layout_height,
                            obstacles=list(WALL_INDEXES))
            for row, col in nearest_food_directions:
                FOOD_CHANNEL[row][col] = OBSERVABLE_EXIST

    '''
    Calls dist_utils to generate a distance matrix to pacman then marks the directions nearest to
    far away ghosts as OBSERVABLE_EXIST
    '''
    def set_nearest_ghost_directions():
            pacman_location = convert_coords_to_matrix_indexes(PACMAN_POSITION[0], PACMAN_POSITION[1])
            nearest_ghost_directions = get_closest_cardinals(pacman_location,
                            list(GHOST_INDEXES), layout_width, layout_height,
                            obstacles=list(WALL_INDEXES))
            for row, col in nearest_ghost_directions:
                GHOSTS_DIST_CHANNEL[row][col] = OBSERVABLE_EXIST

    '''
    Creates the contents for all channels
    @param bounded: (Optional) {Boolean} True when pacman has limited vision. Defaults to False
    '''
    def create_channel_matrices(bounded=False):
        # For Pacman: set absolute coordinates
        matrix_row, matrix_col = convert_coords_to_matrix_indexes(PACMAN_POSITION[0], PACMAN_POSITION[1])
        PACMAN_CHANNEL[matrix_row][matrix_col] = OBSERVABLE_EXIST
        if bounded:
            # For Food, Walls and Capsules
            for x, y in VISIBLE_COORDS:
                set_matrix_values(x, y)
            # When there is no food found in visible range
            if not any(OBSERVABLE_EXIST in row for row in FOOD_CHANNEL):
                set_nearest_food_directions()
            # For Ghosts and Scared Ghosts
            set_ghost_matrices(bounded=True)
            # For the Ghost Distance Matrix
            set_nearest_ghost_directions()  # must be after ghost matrix
        else:
            # For Food, Walls and Capsules
            for y in range(layout_height):
                for x in range(layout_width):
                    set_matrix_values(x, y)
            # For Ghosts and Scared Ghosts
            set_ghost_matrices()

    if is_partially_observable:
        bounding_box = derive_observable_coordinates()
        initialise_observable_zones()    # sets all observable cells to OBSERVABLE_EMPTY
        create_channel_matrices(bounded=True)
    else:
            OBSERVABLE_EMPTY = 0
            create_channel_matrices()

    ############################## PREPARE OBSERVATIONS ###############################
    # Create observation matrix as a combination of
    # wall, pacman, ghost, food and capsule matrices

    # Set channel matrices to CNN input observations
    if is_partially_observable and is_sliced:
        observation = np.zeros((NUM_CHANNELS, channel_height, channel_width))
        observation[0] = slice_channel(WALLS_CHANNEL,           bounding_box=bounding_box)
        observation[1] = slice_channel(PACMAN_CHANNEL,          bounding_box=bounding_box)
        observation[2] = slice_channel(GHOSTS_CHANNEL,          bounding_box=bounding_box)
        observation[3] = slice_channel(SCARED_GHOSTS_CHANNEL,   bounding_box=bounding_box)
        observation[4] = slice_channel(FOOD_CHANNEL,            bounding_box=bounding_box)
        observation[5] = slice_channel(CAPSULES_CHANNEL,        bounding_box=bounding_box)
        # observation[6] = slice_channel(GHOSTS_DIST_CHANNEL,     bounding_box=bounding_box)
    else:
        observation = np.zeros((NUM_CHANNELS, layout_height, layout_width))
        observation[0] = WALLS_CHANNEL
        observation[1] = PACMAN_CHANNEL
        observation[2] = GHOSTS_CHANNEL
        observation[3] = SCARED_GHOSTS_CHANNEL
        observation[4] = FOOD_CHANNEL
        observation[5] = CAPSULES_CHANNEL
        # observation[6] = GHOSTS_DIST_CHANNEL

    return observation
