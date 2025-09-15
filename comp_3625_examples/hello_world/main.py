from environment import MiningGame
import numpy as np
import matplotlib.pyplot as plt

# instantiate the game
game = MiningGame(n_mines=2, render=False)
step_max = 1000
cumulative_reward = 0

# array of values
values = np.ones(2)
value_history = np.zeros((step_max, 2))

# learning rate
alpha = 0.02

# repeatedly ask user to choose a place to mine, and execute their choice
for step in range(step_max):
    choice = values.argmax()
    
    reward = game.choose_mine(int(choice))
    cumulative_reward += reward
    print(f"reward: {reward}, cumulative reward: {cumulative_reward}")

    # update perceived values
    values[choice] += alpha * (reward - values[choice])
    value_history[step] = values
    
# print the true reward probabilities for mines 0 and 1
print('true, average values for mine 0/1:', game.get_average_values())

plt.plot(value_history)
plt.legend(['mine0', 'mine1'])
plt.show()