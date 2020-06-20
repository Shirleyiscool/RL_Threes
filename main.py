from agent import Agent
from env import Threes
from train import print_demo_game_stats, train_qlearning

agent = Agent(Threes)

print("No training:\n Random Mode:")
mean_score_random, max_score_random = print_demo_game_stats(agent, n_games=100, level='Hard', mode='random')
print('Greedy Mode:')
mean_score_max, max_score_max = print_demo_game_stats(agent, n_games=100, level='Hard', mode='max')

print("\nQ-learning:")
agent, mean_score_learning, max_score_learning = train_qlearning(agent, n_games=100, n_episodes=1000,
                                                                 n_training_blocks=5, level='Hard')
