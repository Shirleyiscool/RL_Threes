from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential

from agent import *
from env import *

# Define the actions
action_map = {0: 'a', 1: 'w', 2: 's', 3: 'd'}
key_list = list(action_map.keys())
val_list = list(action_map.values())
checkpoint_path = "training_model/dqn-{epoch:04d}"


class ExperienceReplay:
    """
    Store the agent's experiences in order to collect enough
    example to get a reward signal.
    Reference: "https://github.com/brianspiering/rl-course/blob/
    master/labs/lab_5_basket_catch/lab_5_basket_catch.ipynb"
    """

    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, states, game_over):
        self.memory.append([states, game_over])

        # If memory is too large, then evict to reduce memory size
        if len(self.memory) > self.max_memory:
            # Evict oldest
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        # Dimension of the output layer of NN
        n_actions = model.layers[1].output_shape[1]
        # Dimension of the input layer of NN
        env_dim = model.layers[0].input_shape[1]

        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], n_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_continue = self.memory[idx][1]
            # train_x: st
            inputs[i:i + 1] = state_t
            # Initialize target y:
            # q values Q(st, a0), Q(st, a1), Q(st, a2) for each action
            targets[i] = model.predict(state_t)[0]
            # Next max q value for the next state: maxQ(stp1, a*)
            q_sa = np.max(model.predict(state_tp1))
            if game_continue:
                # Update Q(st,at) with Q-learning
                targets[i, action_t] = reward_t + q_sa
            else:
                targets[i, action_t] = reward_t
        return inputs, targets


def hash_num(state):
    """A function to encode the state to an array with the shape of (1,16)"""
    return np.array([[int(i) for row in state for i in row]])


def dqn_train(epsilon=0.1, n_episodes=1001):
    """Train DQN models and save them"""
    # Build a NN model
    model = Sequential()
    model.add(InputLayer(input_shape=(16,)))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy")
    print(model.summary())

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=2048)
    results = []
    model_list = []
    loss = float('inf')

    for e in range(1, n_episodes):
        # The next new episode
        game_threes = Threes()
        # While the game is not over
        while game_threes.playable():

            # Get initial input (as vector)
            current_state = game_threes.state
            allowed_actions = allowed_moves(current_state)

            # Get next action - You guessed it eplison-greedy
            if np.random.rand() > epsilon:
                # action that can maximize the q value
                q = model.predict(hash_num(current_state))
                action = action_map[np.argmax(q[0])]
            else:
                # random action that can be allowed
                action = np.random.choice(allowed_actions)
            # in case the max action is not allowed by the game
            if action not in allowed_actions:
                action = select_best_move_(game_threes)

            # Apply action, get rewards and new state.
            next_state, reward = try_move(current_state, action)
            game_threes.make_move(action)

            # Encode the action
            action_num = key_list[val_list.index(action)]

            # Store experience.
            exp_replay.remember([hash_num(current_state), action_num,
                                 reward, hash_num(next_state)],
                                game_threes.playable())

            # Get collected data to train model.
            inputs, targets = exp_replay.get_batch(model, batch_size=50)

            # Train model on experiences.
            loss = model.train_on_batch(inputs, targets)

        results.append(get_score(game_threes.state))
        if (e == 1) or (e % 5 == 1):
            print(
                f"Epoch: {e:03d}/{n_episodes:,} | Loss value: {loss:>6.3f} | "
                f"Mean score: {np.mean(results)} | "
                f"Max score: {np.max(results)}")
            model_list.append(checkpoint_path.format(epoch=e))
            model.save(checkpoint_path.format(epoch=e))
    return model, model_list


def print_dqn_game_states(model, n_games=100, level='Hard'):
    """
    Print the result(mean score and max score) of
    playing demo game with dqn model
    """
    results = []
    for _ in range(n_games):
        # A new game
        new_game = Threes(level=level)

        # Play the game till end
        while new_game.playable():
            current_state = new_game.state
            allowed_actions = allowed_moves(current_state)
            action_num = model.predict(hash_num(current_state))
            # Get the best action based on dqn model
            action = action_map[np.argmax(action_num[0])]
            # In case the best action is not allowed
            if action not in allowed_actions:
                action = select_best_move_(new_game)
            new_game.make_move(action)

        # Add the game result to the list
        results.append(get_score(new_game.state))

    mean_score, max_score = np.mean(results), np.max(results)
    print(f"mean score: {mean_score} | max score: {max_score} ")
    return mean_score, max_score
