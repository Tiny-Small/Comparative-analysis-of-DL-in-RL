# Replay memory
from collections import deque
import numpy as np
import random

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.sample_probs = None  # Store probabilities to avoid recalculating
        self.max_priority = 1.0  # Start with a default max priority of 1
        self.last_priority_scale = None

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
        self.sample_probs = None  # Invalidate cached probabilities

    # priority_scale = 0 is uniform sampling
    def get_probabilities(self, priority_scale):
        if self.sample_probs is None or priority_scale != self.last_priority_scale:
            scaled_priorities = np.array(self.priorities) ** priority_scale
            self.sample_probs = scaled_priorities / sum(scaled_priorities)
            self.last_priority_scale = priority_scale
        return self.sample_probs

    # get_importance corrects the bias introduced by prioritised sampling
    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * (1/probabilities)
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = [self.buffer[idx] for idx in sample_indices]

        # Calculate importance weights for the sampled experiences
        importance = self.get_importance(np.array(sample_probs)[sample_indices])
        return samples, importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        np_priorities = np.array(self.priorities)
        errors = np.abs(errors) + offset
        np_priorities[indices] = errors
        self.max_priority = np.max(np_priorities)
        self.priorities = deque(np_priorities, maxlen=self.buffer.maxlen)
        self.sample_probs = None  # Invalidate cached probabilities after updating priorities
