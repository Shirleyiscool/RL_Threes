from train_ql import print_demo_game_stats, train_qlearning
from tensorflow.keras.models import load_model
from dqn import *
import logging


logging.getLogger('tensorflow').disabled = True

# Baseline performance
print("No training:\n Random Mode:")
agent = Agent(Threes)
mean_score_random, max_score_random = print_demo_game_stats(agent, n_games=1000, level='Hard', mode='random')

# Simple DP method
print('Greedy Mode:')
agent = Agent(Threes)
mean_score_max, max_score_max = print_demo_game_stats(agent, n_games=1000, level='Hard', mode='max')

# Q-learning methods
print("\nQ-learning:")
agent = Agent(Threes)
agent, mean_score_learning, max_score_learning = train_qlearning(agent, n_games=1000, n_episodes=1000,
                                                                 n_training_blocks=150, level='Hard')

# Train and test DQN Model
print("\n Train dqn model:")
model, model_list = dqn_train(epsilon=0.1, n_episodes=1001)
last_model = load_model(model_list[-1])
print_dqn_game_states(last_model)
