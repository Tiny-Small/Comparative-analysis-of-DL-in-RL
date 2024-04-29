import numpy as np

def prepare_observation_matrix(state, NUM_CHANNELS, channel_height, channel_width,
            VISIBLE_LIMIT, AUDIBLE_LIMIT, CHANNEL_RADIUS,
            is_partially_observable, is_sliced):
    layout_height = state.data.layout.height
    layout_width = state.data.layout.width
    channel_height = 7
    channel_width = 7  # Set both height and width to 7 for a 7x7 grid

    def initialise_channel_matrix():
        return np.zeros((layout_height, layout_width), dtype=float)

    def convert_coords_to_matrix_indexes(x, y):
        return layout_height - 1 - y, x

    def slice_channel_around_pacman(channel_matrix, pacman_position):
        x, y = pacman_position
        row, col = convert_coords_to_matrix_indexes(x, y)

        # Calculate start and end indices for slicing to ensure 7x7 grid
        half_size = channel_width // 2
        start_row = max(0, row - half_size)
        end_row = min(layout_height, row + half_size + 1)
        start_col = max(0, col - half_size)
        end_col = min(layout_width, col + half_size + 1)

        # Adjust start and end if near borders
        if (end_row - start_row) < channel_height:
            if start_row == 0:
                end_row = min(layout_height, start_row + channel_height)
            else:
                start_row = max(0, end_row - channel_height)
        if (end_col - start_col) < channel_width:
            if start_col == 0:
                end_col = min(layout_width, start_col + channel_width)
            else:
                start_col = max(0, end_col - channel_width)

        # Slice the matrix to the computed indices
        return channel_matrix[start_row:end_row, start_col:end_col]

    # Initialize channel matrices
    walls_channel = initialise_channel_matrix()
    pacman_channel = initialise_channel_matrix()
    ghosts_channel = initialise_channel_matrix()
    scared_ghosts_channel = initialise_channel_matrix()
    food_channel = initialise_channel_matrix()
    capsules_channel = initialise_channel_matrix()

    # Fill channels based on state data
    for x in range(layout_width):
        for y in range(layout_height):
            row, col = convert_coords_to_matrix_indexes(x, y)
            walls_channel[row, col] = 1 if state.data.layout.walls[x][y] else 0
            food_channel[row, col] = 1 if state.data.food[x][y] else 0

    capsules = {(x, y) for x, y in state.data.capsules}
    for x, y in capsules:
        row, col = convert_coords_to_matrix_indexes(x, y)
        capsules_channel[row, col] = 1

    pacman_pos = state.getPacmanPosition()
    row, col = convert_coords_to_matrix_indexes(pacman_pos[0], pacman_pos[1])
    pacman_channel[row, col] = 1

    for ghost_state in state.data.agentStates:
        if not ghost_state.isPacman:
            pos = ghost_state.configuration.getPosition()
            row, col = convert_coords_to_matrix_indexes(int(pos[0]), int(pos[1]))
            channel = scared_ghosts_channel if ghost_state.scaredTimer > 0 else ghosts_channel
            channel[row, col] = 1

    # Slice each channel around Pacman position
    observation = np.zeros((NUM_CHANNELS, channel_height, channel_width))
    observation[0] = slice_channel_around_pacman(walls_channel, pacman_pos)
    observation[1] = slice_channel_around_pacman(pacman_channel, pacman_pos)
    observation[2] = slice_channel_around_pacman(ghosts_channel, pacman_pos)
    observation[3] = slice_channel_around_pacman(scared_ghosts_channel, pacman_pos)
    observation[4] = slice_channel_around_pacman(food_channel, pacman_pos)
    observation[5] = slice_channel_around_pacman(capsules_channel, pacman_pos)

    return observation
