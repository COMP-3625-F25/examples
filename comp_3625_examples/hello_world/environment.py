import numpy as np
import matplotlib.pyplot as plt
from imageio.v3 import imread
import os


class MiningGame:
    """
    A very simple game: choose from two mines and receive a reward. Deduce which mine gives better reward overall
    """

    def __init__(self, n_mines=2, render_delay=0.001, render=True):
        assert n_mines >= 2
        self.n_mines = n_mines
        self.reward_probabilities = np.array([0.4, 0.6] + [0.5] * (n_mines-2))
        np.random.shuffle(self.reward_probabilities)
        dirname = os.path.dirname(__file__)
        self.im = np.hstack([imread(f'{dirname}/images/img_{i % 2}.png') for i in range(n_mines)])
        self.render_delay = render_delay
        self.render = render
        if self.render:
            self._render()

    def get_average_values(self):
        return self.reward_probabilities + -1 * (1 - self.reward_probabilities)

    def choose_mine(self, mine_number: int):
        """
        select a place to mine
        :param mine_number: mine number 0 or 1
        :return: reward obtained
        """
        reward = -1
        if np.random.random() < self.reward_probabilities[mine_number]:
            reward = 1

        if self.render:
            self._render(mine_number, reward)
        return reward

    def _render(self, choice: int = None, reward: float = None):
        plt.cla()

        plt.imshow(self.im)
        plt.xticks([])
        plt.yticks([])

        for i in range(self.n_mines):
            plt.text(x=0 + 900 * i, y=10, s=f"mine {i}", c='w', va='top')

        if reward is not None and choice is not None:
            plt.text(x=np.random.randint(100, 600) + choice * 900,
                     y=np.random.randint(100, 900),
                     s=f'${"+" if reward > 0 else ""}{reward}',
                     fontsize=20, fontweight='bold', backgroundcolor='w',
                     c='r' if reward < 0 else 'y')

        plt.pause(self.render_delay)

