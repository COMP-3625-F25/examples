import numpy as np

class StoreInventory:
    """ 
    a silly store inventory MDP
    agent has to buy stock to avoid going into backorder, but there's a cost to holding stock in inventory    
    """

    def __init__(self):
        self.max_inventory = 10
        self.order_cost = -1
        self.holding_cost = -0.2
        self.backorder_cost = -2

        self.current_inventory = 5

    def get_current_state(self):
        return self.current_inventory
    
    def get_actions(self, state=None):
        return ['buy', 'wait']

    def step(self, action: str) -> tuple[float, float]:

        # how many units purchased by customers today?
        purchases = np.random.choice([0, 1, 2, 3], p=[0.4, 0.4, 0.1, 0.1])
        print('customer purchases:', purchases)

        # did the agent order a new unit?
        orders = 1 if action == 'buy' else 0

        # current inventory
        self.current_inventory = max(-self.max_inventory, min(self.max_inventory, self.current_inventory - purchases + orders))

        # calculate costs
        reward = orders * self.order_cost
        if self.current_inventory < 0:
            reward -= self.current_inventory * self.backorder_cost
        else:
            reward += self.current_inventory * self.holding_cost
        
        return reward, self.get_current_state()


if __name__ == '__main__':

    mdp = StoreInventory()

    for step in range(1000):
        state = mdp.get_current_state()
        print('current inventory:',state)
        action = input(f'choose action from {mdp.get_actions()}: ')
        r, new_state = mdp.step(action)
        print(f'reward: ${r}')