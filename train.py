import numpy as np


def print_demo_game_stats(agent, n_games=100, level='Hard', mode='random'):
    """print the result(mean score and max score) of playing demo game for some times"""
    print(f'mode: {mode}')
    results = [agent.demo_game(level=level, mode=mode) for _ in range(n_games)]
    mean_score, max_score = np.mean(results), np.max(results)
    print(f"mean score: {mean_score}")
    print(f"max score: {max_score}")


def train_qlearning(agent, n_games=100, n_training_blocks=10, level='Hard'):
    """Given agent, do more training. Return (hopefully) improved agent."""
    print("Before learning:")
    print_demo_game_stats(agent, n_games=n_games, level=level, mode='random')
    for n_training_block in range(1, n_training_blocks + 1):
        agent.learn_game(n_games)
        print(f"After {n_games * n_training_block:,} learning games:")
        print_demo_game_stats(agent, n_games=n_games, level=level, mode='q-learning')
    return agent
