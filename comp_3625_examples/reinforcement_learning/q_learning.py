import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sequential_decisions.environments import RusselNorvigMDP
import numpy as np
import random
import time


# instantiate the MDP
mdp = RusselNorvigMDP()
state, _ = mdp.reset()

# algorithm parameters
r_max = 1
alpha = 0.1
gamma = 0.9

# set up a Q table
# initially it only contains values for the start state (that's the only state the agent knows about right now)
# all values initialized to 1 (optimistic initialization)
Q = {state: {act: r_max for act in mdp.get_actions(state)}}


for step in range(1000):

    # lookup action Q values for this state - choose the action with the highest Q value
    action = max(Q[state], key=Q[state].get)

    # execute the Q value in the environment
    new_state, reward, terminated, _, _ = mdp.step(action)

    # add the new state to the Q table if it's never been seen before
    if new_state not in Q:
        Q[new_state] = {act: r_max for act in mdp.get_actions(state)}

    # UPDATE Q TABLE HERE
    

    # render the environment
    mdp.render()
    
    # prepare for next step
    if terminated:
        state, _ = mdp.reset()
    else:
        state = new_state
    
    # small delay to make movements easier to see
    time.sleep(0.01)