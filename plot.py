import matplotlib.pyplot as plt
from deep_rl import *

if __name__ == "__main__":

    plotter = Plotter()
    dirs = [
        'dqn_pixel_atari-181030-150008' 
    ]
    names = [
        'DQN'
    ]

    plt.figure()
    for i, dir in enumerate(dirs):
        data = plotter.load_results(['./log/%s' % (dir)], episode_window=100)
        x, y = data[0]
        plt.plot(x, y, label=names[i])
    plt.xlabel('steps')
    plt.ylabel('episode return')
    plt.legend()

    plt.savefig("DQN.png")




