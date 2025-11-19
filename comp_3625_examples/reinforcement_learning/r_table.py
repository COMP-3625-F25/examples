

class Rtable:

    def __init__(self):
        self.rewards = dict()
    
    def update(self, state, action, new_state, reward) -> None:
        """ updates internal rewards dict to reflect this new information """
        key = state, action, new_state

        if key not in self.rewards:
            self.rewards[key] = (reward, 1)
        else:
            self.rewards[key][1] += 1
            self.rewards[key] = self.rewards[key] + ( reward - self.rewards[key]) / self.transtion_count


    def get_reward(self, state, action, new_state) -> float:
        """ return average reward for given transition """
        reward = self.rewards[(state, action, new_state)]
        
        if self.rewards[(state,action,new_state)] not in self.rewards:
            reward = -1
        return reward
