import matplotlib.pyplot as plt
from environments import RusselNorvigMDP


def process(event):
    if event.key == 'up':
        action = 0
    elif event.key == 'right':
        action = 1
    elif event.key == 'down':
        action = 2
    elif event.key == 'left':
        action = 3
    state, reward, terminated, _, _ = mdp.step(action)
    print('reward:', reward, 'new state:', state)
    mdp.render()
    if terminated:
        print('done')
        plt.pause(1)
        exit(0)

mdp = RusselNorvigMDP()
fig, ax = plt.subplots(1, 1)
fig.canvas.mpl_connect('key_press_event', process)
mdp.render()

plt.show()