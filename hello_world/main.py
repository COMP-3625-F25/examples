from environment import MiningGame
import numpy as np
import matplotlib.pyplot as plt

# instantiate the game
game = MiningGame(n_mines=2, render=True)
step_max = 10
cumulative_reward = 0

# repeatedly ask user to choose a place to mine, and execute their choice
for step in range(step_max):
    choice = input("choose a place to mine (0 or 1). q to quit: ")
    if choice == 'q':
        break
    
    reward = game.choose_mine(int(choice))
    cumulative_reward += reward
    print(f"reward: {reward}, cumulative reward: {cumulative_reward}")
    
# print the true reward probabilities for mines 0 and 1
print('true, average values for mine 0/1:', game.get_average_values())
