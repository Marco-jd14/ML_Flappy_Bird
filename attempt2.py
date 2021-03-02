import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'FlappyBird-v0' in env or 'FlappyBird-rgb-v0' in env:
        del gym.envs.registration.registry.env_specs[env]
import flappy_bird_gym

import matplotlib.pyplot as plt
import time
import pygame
from argparse import ArgumentParser
import numpy as np
from datetime import datetime
np.set_printoptions(suppress=True)


nr_feats = 3

def main(options):

    for i in range(10):
        info = play_game(options.verbose, options.show_gui, options.fps)


    # Minima: [  0.00347222  -0.51070313     -8.        ]
    # Maxima: [  1.64236111   0.52148438    380.48      ]

    h_range = np.linspace(0.0, 1.65, 1000)
    v_range = np.linspace(-0.52, 0.53, 1000)

    h_states = np.zeros_like(h_range, dtype=int)
    v_states = np.zeros_like(v_range, dtype=int)

    for i in range(len(h_range)):
        h_states[i] = h_state(h_range[i])
    for i in range(len(v_range)):
        v_states[i] = v_state(v_range[i])

    print(len(set(h_states)))
    print(len(set(v_states)))


    # env = flappy_bird_gym.make("FlappyBird-v0")
    # q_table = np.zeros([env.observation_space.n, env.action_space.n])

def h_state(h_dist):
    # states 0-99
    min_value = 0
    max_value = 0.6

    if h_dist < min_value:
        return 0
    elif h_dist > max_value:
        return 99

    #first scale between 0 and 1 then distribute over the remaining nr of states
    return int((h_dist-min_value)/(max_value-min_value) * (100-2))

def v_state(v_dist):
    # states 0-99
    min_value = -0.10
    max_value = 0.10

    if v_dist < -0.3:
        return 0
    elif v_dist < -0.2:
        return 1
    elif v_dist < -0.15:
        return 2
    elif v_dist < min_value:
        return 3
    if v_dist > 0.3:
        return 99
    elif v_dist > 0.2:
        return 98
    elif v_dist > 0.15:
        return 97
    elif v_dist > max_value:
        return 96

    #first scale between 0 and 1 then distribute over the remaining nr of states
    return int((v_dist-min_value)/(max_value-min_value) * (100-8)) + 4

def play_game(show_prints=False, show_gui=False, fps=100):

    env = flappy_bird_gym.make("FlappyBird-v0")
    obs = env.reset()

    if show_gui:
        env.render()

    prev_score = -1
    prev_sec = -1
    while True:
        if show_gui:
            pygame.event.pump()

        obs = env._get_observation()
        if obs[1] < -0.05:
            action = 1 #flap
        else:
            action = 0 #idle
        # action = 1

        # Processing:
        obs, reward, done, info = env.step(action)

        if show_prints:
            # if prev_score != info['score']:
            # now = datetime.now().second
            # if prev_sec != now:
            if reward > 0:
            #     prev_sec = now
                # prev_score = info['score']

                print(obs)
                print("\tReward:", reward, "\tdied:",done, "\tinfo:",info)

        # Rendering the game:
        # (remove this two lines during training)
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

        # Checking if the player is still alive
        if done:
            break

    env.close()

    return info






if __name__=='__main__':
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
                        default=100,
                        help="Specify in how many FPS the game should run")

    options = parser.parse_args()

    main(options)