from collections import deque

class VisitedTilesTracker:
    def __init__(self, stalling_threshold=20, no_new_position_threshold=10):
        self.visited_positions = set()
        self.position_history = deque(maxlen=stalling_threshold)
        self.no_new_position_threshold = no_new_position_threshold
        self.steps_since_last_new_position = 0

    def visit(self, position):
        is_new_visit = position not in self.visited_positions
        if is_new_visit:
            self.visited_positions.add(position)
            self.steps_since_last_new_position = 0  # Reset counter on new visit
        else:
            self.steps_since_last_new_position += 1  # Increment counter if position is not new

        self.position_history.append(position)  # Keep adding position to track stalling in a small area
        is_stalling = self.detect_stalling() or self.steps_since_last_new_position >= self.no_new_position_threshold

        return is_new_visit, is_stalling, len(self.visited_positions)

    def detect_stalling(self):
        # Detecting stalling based on lack of movement in a small area
        if len(self.position_history) < self.position_history.maxlen:
            return False  # Not enough history to determine stalling
        recent_area = set(self.position_history)
        return len(recent_area) <= (self.position_history.maxlen / 2)

    def reset(self):
        self.visited_positions.clear()
        self.position_history.clear()
        self.steps_since_last_new_position = 0



class PelletConsumptionTracker:
    def __init__(self, gameState):
        self.last_pellet_position = None
        self.consecutive_pellets = 0
        # Store the initial state of the food grid
        self.previousFood = gameState.getFood().deepCopy()

    def reset(self, gameState):
        self.last_pellet_position = None
        self.consecutive_pellets = 0
        self.previousFood = gameState.getFood().deepCopy()

    def check_pellet_consumption(self, position, gameState):
        x, y = position
        currentFood = gameState.getFood()

        # Pellet is consumed if Pac-Man moved to a new position that had food
        # in the previous state but doesn't have food now.
        if self.previousFood[x][y] and not currentFood[x][y]:
            # Check if this is consecutive based on the last known position
            if self.last_pellet_position:
                distance = abs(x - self.last_pellet_position[0]) + abs(y - self.last_pellet_position[1])
                if distance == 1:
                    self.consecutive_pellets += 1
                else:
                    self.consecutive_pellets = 1  # Reset if not consecutive
            else:
                self.consecutive_pellets = 1  # First pellet consumed
            self.last_pellet_position = position
        else:
            # Reset if no pellet was consumed this move
            self.consecutive_pellets = 0

        # Update the previous food state for the next move
        self.previousFood = currentFood.deepCopy()

        return self.consecutive_pellets


class EndGameEvaluator:
    def __init__(self, gameState, max_pellet_bonus = 100):
        # At the start, calculate the total number of pellets in the game
        self.total_pellets = gameState.getNumFood()
        self.max_pellet_bonus = max_pellet_bonus  # Maximum bonus for eating all pellets

    def remaining_pellet_bonus(self, gameState):
        """
        Calculate a bonus based on the number of pellets remaining. The fewer
        pellets remaining, the higher the bonus.
        """
        remaining_pellets = gameState.getNumFood()  # Directly get the number of pellets left in the current game state
        pellets_eaten = self.total_pellets - remaining_pellets
        bonus = (pellets_eaten / self.total_pellets) * self.max_pellet_bonus
        return bonus
