


import numpy as np

class Ttable:

    def __init__(self):
        self.counts = dict()
        self.all_states = set()
    
    def get_transition(self, state, action) -> dict:
        """
        returns a dictionary mapping next_state -> probabilities
        """
        dic = self.counts[(state,action)]
        total = sum(dic.values())
        return_dic = {}
        for state, count in dic.items():
            return_dic[state] = count / total
        return return_dic

    def update(self, state, action, new_state) -> None:
        """
        update the internal counts
        """
        self.all_states.add(state)
        self.all_states.add(new_state)

        if (state, action) not in self.counts:
            self.counts[(state, action)] = {new_state: 1}
        elif new_state not in self.counts[(state, action)]:
            self.counts[(state, action)][new_state] = 1
        else:
            self.counts[(state, action)][new_state] += 1
    
    def get_all_states(self) -> list:
        """ return list of all known states """
        return self.all_states

t = Ttable()
t.update('stateA', 'action1', 'stateB')
t.update('stateA', 'action1', 'stateC')
t.update('stateA', 'action1', 'stateB')
print(t.get_transition('stateA', 'action1'))
print(t.get_all_states())