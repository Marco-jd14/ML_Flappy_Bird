import MarcoCarlo
import attempt2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import flappy_bird_gym

def main():
    repeat = 10
    env = flappy_bird_gym.make("FlappyBird-v0")

    filename = "q_table.npy"
    with open(filename, 'rb') as f:
        q_table = np.load(f)

    start = datetime.now()
    results = np.zeros(repeat)
    for i in range(repeat):
        # results[i] = MarcoCarlo.play_game(env, show_prints=False, show_gui=False)['score']
        results[i] = attempt2.play_q_game(q_table, env, show_prints=False, show_gui=False)['score']

    end = datetime.now()

    fig, axs = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.8)

    axs[0].plot(results, 'r', linewidth=0.7)
    axs[0].set_title("No learning involved")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Points scored")
    axs[0].set_xlim([0,repeat])

    axs[1].hist(results, bins=50, color='red')
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlabel("Points scored")
    axs[1].set_title("Took %.2f mins"%((end-start).seconds/60))
    axs[1].set_xlim(0)

    plt.savefig("test.png")

if __name__=='__main__':
    main()