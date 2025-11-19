

class Ttable:

    def __init__(self):
        self.counts = dict()
    
    def get_transition(self, state, action) -> dict:
        """
        returns a dictionary mapping next_state -> probabilities
        """
        dic = self.counts[(state,action)]
        sum = np.sum(dic.values())
        return {key: (value / sum) for key, value in dic.items()}

    def update(self, state, action, new_state) -> None:
        """
        update the internal counts
        """
        pass
    
    def get_all_states(self) -> list:
        """ return list of all known states """
        pass

