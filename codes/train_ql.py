import numpy as np


def print_demo_game_stats(agent, n_games=100, level='Hard', mode='random'):
    """print the result(mean score and max score) of playing demo game"""
    results = [agent.demo_game(level=level, mode=mode) for _ in range(n_games)]
    mean_score, max_score = np.mean(results), np.max(results)
    print(f"mean score: {mean_score} | max score: {max_score} ")
    return mean_score, max_score


def train_qlearning(agent, n_games=100, n_episodes=100,
                    n_training_blocks=10, level='Hard'):
    """Given agent, do more training. Return (hopefully) improved agent."""
    for n_training_block in range(1, n_training_blocks + 1):
        agent.learn_game(n_episodes)
        print(f"After {n_episodes * n_training_block:,} learning games:")
        mean_score, max_score = \
            print_demo_game_stats(agent, n_games=n_games,
                                  level=level, mode='q-learning')
    return agent, mean_score, max_score
