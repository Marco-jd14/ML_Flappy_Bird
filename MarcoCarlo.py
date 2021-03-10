import time
import pygame
import flappy_bird_gym
from argparse import ArgumentParser


def start(show_prints=False, show_gui=False, fps=60, agents_num=1):
    # create gym
    env = flappy_bird_gym.make("FlappyBird-v0")
    # get initial observation
    obs = env.reset(agents_num)

    # if GUI enabled, initialise the window and clear events
    if show_gui:
        env.render()
        pygame.event.pump()

    # Training
    while True:
        # make decision
        action = obs[0][1] < -0.05

        # Processing:
        obs, reward, done, scores = env.step([action, 1, 0])

        # logging
        if show_prints:
            print("")
            for i in range(len(obs)):
                print("BIRD %d:\t" % i, obs[i], "\tReward:", reward[i], "\tdied:", done[i], "\tinfo:", scores[i])

        # Rendering the game:
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

        # Checking if any the player is still alive
        if all(done):
            break

    # clean up
    env.close()
    return scores


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-g",
                        dest="show_gui",
                        action="store_true",
                        help="Whether the game GUI should be shown or not")

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        action="store_true",
                        help="Print information while playing the game")

    parser.add_argument("-fps",
                        dest="fps",
                        type=int,
                        default=60,
                        help="Specify in how many FPS the game should run")

    parser.add_argument("-n",
                        dest="agents_num",
                        type=int,
                        default=1,
                        help="Specify in how many agents the game should has")

    options = parser.parse_args()

    start(show_prints=options.verbose, show_gui=options.show_gui, fps=options.fps, agents_num=options.agents_num)
