import itertools
from gymnasium import Env
import numpy as np
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt
from typing import Tuple, TypeVar
import random
ObsType = TypeVar("ObsType")


class ToyMDP(Env[ObsType, int]):

    def __init__(self, transition_probabilities, reward_function, start_state=None):
        """
        :param transition_probabilities: nested dicts of state -> action -> new_state -> probability
        :param reward_function: callable with signature:
                function(old_state: ObsType, action: int, new_state: ObsType) -> float
        :param current_state: the start state for the agent
        """
        self.t = transition_probabilities
        self.r = reward_function
        self.start_state = start_state
        self.current_state = start_state or np.random.choice(list(self.t))
        self.action_space = Discrete(len(set(itertools.chain.from_iterable([list(self.t[state]) for state in self.t]))))
        self.observation_space = Discrete(len(self.t))

    def all_states(self):
        all_s = set(self.t)
        # find and add terminal states
        for _, a in self.t.items():
            for s in a.values():
                all_s.update(s)
        return all_s


    def get_actions(self, state=None) -> list:
        """
        get actions available from state
        :param state: the state
        :return: list of available actions
        """
        return list(self.t.get(state or self.current_state, []))

    def get_transition_probabilities(self, state, action):
        """
        get the distribution over next states, when action is executed from state
        :param state: the state the agent is in
        :param action: the action executed from state
        :return: dict mapping next-state -> probability
        """
        if state in self.t and action in self.t[state]:
            return self.t[state][action]
        return {}

    def get_reward(self, state, action, new_state) -> float:
        """
        get the reward for a given state, action, new_state transition
        :param state: the original state
        :param action: the action executed from state
        :param new_state: the new state encountered after taking action
        :return: a sample of the reward for this transition
        """
        return self.r(state, action, new_state)

    def _get_observation(self) -> ObsType:
        return self.current_state

    def reset(self, **kwargs) -> tuple[ObsType, dict]:
        """
        randomize state and return observation
        :return: observation for the new state
        """
        self.current_state = self.start_state or random.choice(list(self.t))
        return self._get_observation(), dict()

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, dict]:
        """
        execute one step in the environment
        :param action: action number to perform
        :return: new state, reward, terminated, and default values for truncated, info, and done
        """
        # get dict of possible new states & probabilities
        possible_new_states = self.t[self.current_state][action]

        # randomly choose next state, per the probabilities
        new_state = list(possible_new_states.keys())[np.random.choice(len(possible_new_states), p=list(possible_new_states.values()))]

        # decide reward
        reward = self.r(self.current_state, action, new_state)

        # decide terminated
        terminated = new_state not in self.t

        self.current_state = new_state
        return self._get_observation(), reward, terminated, False, {}

    def render(self):
        pass

    def close(self):
        pass


class TwoStateMDP(ToyMDP[tuple]):
    """
    a tiny MDP with two states and two actions
    """

    def __init__(self, **kwargs):
        # dict of (state, action, new_state) -> reward
        self.reward_mean_std = {
            ('B', 1, 'A'): (10, 0.5),
            ('A', 2, 'B'): (-1, 0.5),
            ('A', 1, 'A'): (2, 0.5)
        }

        super().__init__(
            transition_probabilities={
                'A': {
                    1: {'A': 1.0},
                    2: {'A': 0.2, 'B': 0.8}
                },
                'B': {
                    1: {'A': 1.0},
                    2: {'B': 1.0}
                }
            },
            reward_function=lambda s, a, s_prime: np.random.normal(*self.reward_mean_std.get((s, a, s_prime), (0, 0)))
        )

class ThreeStateMDP(ToyMDP[tuple]):
    """
    An implementation of the small markov decision process illustrated at
    https://en.wikipedia.org/wiki/Markov_decision_process#/media/File:Markov_Decision_Process.svg
    """

    def __init__(self, **kwargs):

        # dict of (state, action, new_state) -> reward
        self.rewards = {
            (1, 0, 0): 5,
            (2, 1, 0): -1
        }

        super().__init__(
            transition_probabilities={
                0: {
                    0: {0: 0.5, 2: 0.5},
                    1: {2: 1.0}
                },
                1: {
                    0: {1: 0.1, 0: 0.7, 2: 0.2},
                    1: {1: 0.95, 2: 0.05}
                },
                2: {
                    0: {2: 0.6, 0: 0.4},
                    1: {2: 0.4, 0: 0.3, 1: 0.3}
                }
            },
            reward_function=lambda s, a, s_prime: self.rewards.get((s, a, s_prime), 0)
        )

    def render(self):
        pass

    def close(self):
        pass


class RusselNorvigMDP(ToyMDP[tuple]):
    """
    simple gridworld MDP from the Russel/Norvig textbook, chapter 17
    """

    def __init__(self, movement_cost=-0.04):
        self.movement_cost = movement_cost
        self.action_directions = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}

        # build the nested dictionaries that define the transition probabilities
        transition_probabilities = {}
        for x in range(1, 5):
            for y in range(1, 4):
                state = (x, y)
                if np.array_equal(state, [4, 3]) or np.array_equal(state, [4, 2]):
                    continue
                transition_probabilities.setdefault(state, {})
                for action in range(4):
                    transition_probabilities[state].setdefault(action, {})
                    for effective_action, p in [(action, 0.8), ((action - 1)%4, 0.1), ((action + 1)%4, 0.1)]:
                        new_state = tuple(np.add(state, self.action_directions[effective_action]))
                        if new_state[0] > 4 or new_state[1] > 3 or np.min(new_state) < 1 or np.array_equal(new_state, [2,2]):
                            new_state = state
                        current_p = transition_probabilities[state][action].get(new_state, 0)
                        transition_probabilities[state][action][new_state] = current_p + p

        super().__init__(
            transition_probabilities=transition_probabilities,
            reward_function=self._reward_function,
            start_state=(1, 1)
        )

    def get_actions(self, state=None):
        return [0, 1, 2, 3]

    def _reward_function(self, old_state, action, new_state):
        if np.array_equal(new_state, [4, 3]):
            return 1
        if np.array_equal(new_state, [4, 2]):
            return -1
        return self.movement_cost

    def render(self):
        # plt.ion()
        grid = np.ones((3, 4)) * 1
        grid[1, 1] = 0
        grid[0, 3] = 0.33
        grid[1, 3] = 0.6

        plt.cla()
        plt.imshow(grid, cmap='cubehelix', norm=None)

        plt.text(x=self.current_state[0]-1, y=2-(self.current_state[1]-1), s='agent', horizontalalignment='center', verticalalignment='center', size='xx-large')

        plt.xticks([0, 1, 2, 3], [1, 2, 3, 4])
        plt.yticks([0, 1, 2], [3, 2, 1])
        plt.pause(0.001)


class TicTacToe(Env):

    def __init__(self):
        self.state = np.zeros(9, dtype=int)
        self._player_turn = 1

    def reset(self, *args) -> Tuple[tuple, dict]:
        """
        resets the game
        :return: current state, None
        """
        self.state = np.zeros(9, dtype=int)
        self._player_turn = 1
        return self._get_obs(), dict()

    def state_string(self, obs=None) -> str:
        """
        returns an ascii printout of the tic-tac-toe board
        :param obs: a board state (as returned by step and reset). defaults to current state
        :return: string
        """
        obs = obs or self.state
        string = f'{obs[0]}│{obs[1]}│{obs[2]}\n' \
                 f'─┼─┼─\n{obs[3]}│{obs[4]}│{obs[5]}\n' \
                 f'─┼─┼─\n{obs[6]}│{obs[7]}│{obs[8]}\n'
        string = string.replace('0', ' ').replace('-1', 'O').replace('1', 'X')
        return string

    def render(self) -> None:
        """
        prints an ascii representation of the tic-tac-toe board
        """
        print(self.state_string())

    def get_actions(self, obs=None) -> list:
        """
        gets currently available actions
        :param obs: a board state (as returned by step and reset). defaults to current state
        :return: list of legal numeric actions
        """
        obs = np.array(obs or self.state)
        return list(np.where(obs == 0)[0])

    def step(self, action) -> Tuple[tuple, float, bool, bool, None]:
        """
        execute an action in the game
        :param action: numeric action to perform (use get_actions() to get a list of currently available actions
        :return: observation, reward, False, False, None
        """
        self.state[action] = self._player_turn
        self._player_turn *= -1

        game_won = self._game_won()
        draw = not any(self.state == 0)
        terminated = bool(game_won) or draw

        reward = 0
        if game_won:
            reward = 1

        return self._get_obs(), reward, terminated, False, None

    def _get_obs(self) -> tuple:
        return tuple(self.state)

    def _game_won(self, obs=None):
        obs = np.array(obs or self.state)
        grid = obs.reshape((3, 3))
        sums = np.concatenate((
            grid.sum(axis=0),
            grid.sum(axis=1),
            [grid.diagonal().sum()],
            [np.fliplr(grid).diagonal().sum()]
        ))
        if any(sums == 3):
            return 1
        if any(sums == -3):
            return -1
        return 0


if __name__ == '__main__':

    env = TwoStateMDP()
    state, _ = env.reset()
    while True:
        action = input(f'state={state}, choose from actions: {env.get_actions(state)} ("q" to quit): ')
        if action == 'q':
            break

        state, reward, _, _, _ = env.step(int(action))
        if reward != 0:
            print(f'got reward of {reward:.2f}')
