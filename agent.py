from env import *


def hashable(state):
    """Switch state matrix to string matrix, so as to make it hashable."""
    return ', '.join([str(int(i)) for row in state for i in row])


def select_best_move_(game):
    """Selects best move that can get the maximum reward for the next state"""
    possible_next_actions = allowed_moves(game.state)
    state_action_score = [(move, try_move(game.state, move)[1])
                          for move in possible_next_actions]
    max_score = max(state_action_score, key=lambda item: item[1])[1]
    max_move_list = [move for move, score in state_action_score if score == max_score]
    best_next_move = np.random.choice(max_move_list)
    return best_next_move


class Agent:
    """
    This is an agent to play game "Threes". There are two main mode to play the game. One is human mode and the other
    is computer mode(demo game). For the computer mode, there are currently three methods to play the game: [
    'random', 'max', 'q-learning'] The functions here are inspired by
    "https://github.com/brianspiering/rl-course/blob/master/labs/lab_4_tic_tac_toe/lab_4_tic_tac_toe.ipynb"
    """

    def __init__(self, threes, epsilon=0.1, alpha=1.0):
        """Initial the Agent."""
        self.V = dict()
        self.NewGame = threes
        self.epsilon = epsilon
        self.alpha = alpha

    def state_value(self, game_state, action):
        """Look up state value. If never seen state, then assume 0."""
        return self.V.get((hashable(game_state), action), 0.0)

    def state_values(self, game_state, actions):
        """Return a dictionary of state-value pair. It is for finding the action that can maximize the q value """
        return dict(((hashable(game_state), action), self.state_value(game_state, action)) for action in actions)

    def learn_game(self, n_episodes=1000):
        """Let's learn through complete experience to get that reward."""
        for e in range(1, n_episodes + 1):
            game = self.NewGame()
            while game.playable():
                action, reward = self.learn_from_move(game)
            self.V[(hashable(game.state), action)] = reward

    def learn_from_move(self, game):
        """The heart of Q-learning."""

        current_state = game.state
        # Select next action with epsilon-greedy method
        selected_move = self.learn_select_move(game)

        # Next state s(t+1) and reward r
        next_state, reward = try_move(current_state, selected_move)

        # Current state Q value Q(s, a)
        old_value = self.state_value(current_state, selected_move)

        # best action a* for the next state with the largest q value Q(st+1, a*)
        next_max_V, next_max_move = self.select_best_move(game, next_state)

        # Q-learning that updates the q-value
        self.V[(hashable(current_state), selected_move)] = (1 - self.alpha) * old_value + self.alpha * (
                    reward + next_max_V)

        game.make_move(selected_move)
        return selected_move, reward

    def learn_select_move(self, game):
        """Exploration and exploitation"""
        if np.random.uniform(0, 1) < self.epsilon:
            selected_action = np.random.choice(allowed_moves(game.state))
        else:
            selected_action = self.select_best_move(game, game.state)[1]
        return selected_action

    def select_best_move(self, game, game_state):
        """Selects best move for given state(Greedy)"""
        state_action_values = self.state_values(game_state, allowed_moves(game_state))
        max_V = max(state_action_values.values())
        max_move = np.random.choice([state_action[1] for state_action, v in state_action_values.items() if v == max_V])
        return max_V, max_move

    def demo_game(self, level='hard', mode='random'):
        """Agent plays with different policies (random/max/q-learning)"""
        game = self.NewGame()
        game.level = level
        while game.playable():
            if mode == 'random':
                next_action = np.random.choice(allowed_moves(game.state))
            elif mode == 'max':
                next_action = select_best_move_(game)
            elif mode == 'q-learning':
                next_action = self.select_best_move(game, game.state)[1]
            else:
                return "No such mode"
            game.make_move(next_action)
        return game.score

    def human_mode(self):
        """Interactive mode"""
        game = self.NewGame()
        level = input('level: easy or hard?')
        game.level = level
        print(game.state)
        while game.playable():
            human_allowed_moves = allowed_moves(game.state) + ['stop']
            human_move = input(f'You can input {human_allowed_moves}')
            if human_move == 'stop':
                return f'Game over! Your score is {game.score}'
            game.make_move(human_move)
            print(game.state)
        return f'Game over! Your score is {game.score}'
