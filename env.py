import numpy as np


def movable_condition(first, second):
    """Define whether two close tile can be merged"""
    return ((first == 0) and (second != 0)) or \
           ((np.any(np.array([first, second]) > 2)) and (first == second)) or \
           ((first, second) == (1, 2)) or \
           ((second, first) == (1, 2))


def can_move_col(array):
    """Check whether an array can be merged."""
    for i in range(3):
        first, second = array[i], array[i + 1]
        if movable_condition(first, second):
            return True
    return False


def allowed_moves(state):
    """Find allowed moves for a certain game state"""
    allowed_actions = []
    # Check whether the agent can swipe up
    if np.any([can_move_col(col) for col in state.T]):
        allowed_actions.append('w')
    # Check whether the agent can swipe down
    if np.any([can_move_col(col[::-1]) for col in state.T]):
        allowed_actions.append('s')
    # Check whether the agent can swipe left
    if np.any([can_move_col(row) for row in state]):
        allowed_actions.append('a')
    # Check whether the agent can swipe right
    if np.any([can_move_col(row[::-1]) for row in state]):
        allowed_actions.append('d')
    return allowed_actions


def try_move_col(array):
    """Return the next state for an array"""
    new_array = array.copy()
    for i in range(3):
        first, second = array[i], array[i + 1]
        if movable_condition(first, second):
            new_array[i] = first + second
            new_array[i + 1:] = np.append(new_array[i + 2:], 0)
            return new_array
        else:
            continue


def get_reward(current_state, next_state):
    """Given the current state and the next state, return the reward for the transition action."""
    reward = 0
    # maximum number gets larger
    reward += (np.max(next_state) - np.max(current_state))
    # more merge
    reward += (np.count_nonzero(next_state == 0) - np.count_nonzero(current_state == 0))
    return reward


def try_move(current_state, action):
    """Given the state and the chosen action, return the next state"""
    next_state = current_state.copy()
    allowed_actions = allowed_moves(current_state)
    if action not in allowed_actions:
        print(f'Can not move {action}')
        return current_state

    # Swipe up
    if action == 'w':
        for i, col in enumerate(current_state.T):
            if can_move_col(col):
                next_state.T[i] = try_move_col(col)
    # Swipe down
    elif action == 's':
        for i, col in enumerate(current_state.T):
            if can_move_col(col[::-1]):
                new_array = try_move_col(col[::-1])
                next_state.T[i] = new_array[::-1]
    # Swipe left
    elif action == 'a':
        for i, col in enumerate(current_state):
            if can_move_col(col):
                next_state[i] = try_move_col(col)
    # Swipe right
    elif action == 'd':
        for i, col in enumerate(current_state):
            if can_move_col(col[::-1]):
                new_array = try_move_col(col[::-1])
                next_state[i] = new_array[::-1]

    elif action == 'stop':
        return current_state, np.sum(current_state)

    reward = get_reward(current_state, next_state)

    return next_state, reward


class Threes:
    """
    This is a simulated environment of game Threes.
    Swipe direction: {left: 'a', right: 'd', up: 'w', down: 's'}.
    There are two levels for this game: ['hard', 'easy'], in which default level is 'hard'.
    """

    def __init__(self, level='hard'):
        """Initialize the game"""
        self.state = np.zeros((4, 4))
        x, y = np.random.choice(4, 2)
        self.state[x, y] = np.random.choice([1, 2])
        self.score = np.sum(self.state)
        self.level = level

    def playable(self):
        """Check whether the game is still playable."""
        if len(allowed_moves(self.state)) != 0:
            return True
        else:
            return False

    def gen_new_tile(self):
        """Generate a new tile after each move."""

        # Basic list of numbers that can be selected
        choice_list = [1, 2, 3]

        # More number can be selected when the maximum number on the grid gets larger
        if np.max(self.state) % 3 == 0:
            max_power = np.int(np.log2(np.max(self.state) / 3))
            choice_list += [3 * 2 ** i for i in range(max_power + 1)]

        # Generate the probabilities for each candidate
        if self.level == 'hard':
            norm_prob = [1 / len(choice_list)] * len(choice_list)
        else:
            prob = [i + 1 for i in range(len(choice_list))][::-1]
            norm_prob = [num / sum(prob) for num in prob]

            # return next number
        return np.random.choice(choice_list, p=norm_prob)

    def make_move(self, action):
        """Given the action, the game goes to the next state"""
        if action == 'stop':
            self.score = np.sum(self.state)
            return self.state, self.score

        self.state = try_move(self.state, action)[0]

        # generate new tile for the current state
        new_tile = self.gen_new_tile()
        loc_0 = np.argwhere(self.state == 0)
        x, y = loc_0[np.random.choice(len(loc_0))]

        # Update the game state and scores
        self.state[x, y] = new_tile
        self.score = np.sum(self.state)
