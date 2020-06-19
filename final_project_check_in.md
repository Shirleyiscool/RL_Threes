# Final Project Check-in

- Project Name: Reinforcement Learning in Game "Threes"

- Finalized Research Question: Can an agent wins higher score than humans in the game "Threes"

- The following working code in GitHub: 
    - A environment: **env.py**
    - An agent that performs random actions in the environment: **main.py** `print_demo_game_stats(agent, n_games=100, level='Hard', mode='random')`
    - An agent that learns based on the environment: **main.py** `train_qlearning(agent, n_games=100, n_training_blocks=5, level='Hard')`

- List of ideas to finish project:
    - Q-learning seems not work well for the game Threes as there are too many states that can happen in a game.
    - Will try DQN next week and see whether it can improve the agent's performance. 