import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt
from store_inventory_game import StoreInventory
import numpy as np


# instantate the store inventory mdp
world = StoreInventory()

# table to store transition history, and supervised learning models (one for each action)
history = pd.DataFrame(columns=['state', 'action', 'reward', 'new_state'])
buy_value_model = DecisionTreeRegressor(max_depth=4)
wait_value_model = DecisionTreeRegressor(max_depth=4)


def train(history, gamma=0.9):
    """
    function that trains the models on the given transition history
    """
    history = history.copy()

    # use the current models to predict value of actions at the new states in the history
    try: 
        history['new_state_buy_value'] = buy_value_model.predict(X=history[['state']])
        history['new_state_wait_value'] = wait_value_model.predict(X=history[['state']]) 
    except NotFittedError:
        history['new_state_buy_value'] = history['new_state_wait_value'] = 0

    # value of a transition is:   r + gamma * max(predicted value of actions at new_state)
    history['value'] = history['reward'] + gamma * np.maximum(history['new_state_buy_value'], history['new_state_wait_value'])
    
    # fit models to predict the value calculated above, given the current state
    buy_value_model.fit(
        X=history.loc[history['action'] == 'buy', ['state']], 
        y=history.loc[history['action'] == 'buy', 'value']
        )
    
    wait_value_model.fit(
        X=history.loc[history['action'] == 'wait', ['state']], 
        y=history.loc[history['action'] == 'wait', 'value']
        )


# loop of agent's experience in the world
for step in range(1000):
    state = world.get_current_state()
    print('current inventory:',state)

    # for this demo, choose actions randomly. In practice agent should use it's models to predict the (Q) values of actions, and select based on that
    action = np.random.choice(['buy', 'wait'], p=[0.7, 0.3])
    print('selected action: ', action)
    r, new_state = world.step(action)
    print(f'reward: ${r}')
    
    # add this experience to the history
    history.loc[len(history)] = [state, action, r, new_state]

    # with some frequency, re-train the models.
    # models should get better and better as history (i.e. training dataset) grows
    if step > 50 and step % 20 == 0:
        train(history)


# print model's predictions of value for different states
results = pd.DataFrame({'state': np.arange(-10, 11)})
results['buy_value'] = buy_value_model.predict(results[['state']])
results['wait_value'] = wait_value_model.predict(results[['state']])
print(results)

