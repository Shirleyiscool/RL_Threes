from agent import Agent
from env import Threes
from train import print_demo_game_stats, train_qlearning

agent = Agent(Threes)

print("No training:\n Random Mode:")
print_demo_game_stats(agent, n_games=100, level='Hard', mode='random')
print('Greedy Mode:')
print_demo_game_stats(agent, n_games=100, level='Hard', mode='max')

print("\nQ-learning:")
train_qlearning(agent, n_games=100, n_episodes=1000, n_training_blocks=5, level='Hard')