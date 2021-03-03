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
import random
import sys
import os.path
from datetime import datetime
np.set_printoptions(suppress=True)


nr_states_h = 100
nr_states_v = 100
nr_feats = 2

def main(options):
    env = flappy_bird_gym.make("FlappyBird-v0")

    # for i in range(10):
    #     info = play_game(env, options.verbose, options.show_gui, options.fps)

    repeat = 0
    overwrite = 1
    filename = "q_table4.npy"
    with open(filename, 'rb') as f:
        q_table = np.load(f)

    if overwrite or not os.path.isfile(filename):
        start = datetime.now()
        q_table = np.zeros([nr_states_h, nr_states_v, env.action_space.n])
        # q_table, all_scores, all_rewards, observation_history = q_learning(env, q_table)
        q_table, all_scores, all_rewards = q_learning(env, q_table)
        np.save(filename, q_table)
        end = datetime.now()
        print("\nTraining took %.2f mins\n"%((end-start).seconds/60))

    if overwrite:
        dec = 10
        selected_scores = np.zeros(int(len(all_scores)/dec))
        for i in range(len(selected_scores)):
            selected_scores[i] = np.max(all_scores[i*dec:(i+1)*dec])

    with open(filename, 'rb') as f:
        q_table = np.load(f)

    # play_q_game(q_table, env, fps=20)
    # print(q_table.shape)
    print(len(q_table[np.all(q_table==0.0,axis=2)]))

    # h_zeros = np.zeros(nr_states_h)
    # v_zeros = np.zeros(nr_states_v)
    # for i in range(nr_states_h):
    #     for j in range(nr_states_v):
    #         if any(q_table[i,j] == 0.0):
    #             h_zeros[i] += 1
    #             v_zeros[j] += 1

    # plt.plot(h_zeros)
    # plt.title("h_dist")
    # plt.figure()
    # plt.plot(v_zeros)
    # plt.title("v_dist")

    repeat = 10000
    start = datetime.now()
    results = np.zeros(repeat)
    for i in range(repeat):
        # results[i] = MarcoCarlo.play_game(env, show_prints=False, show_gui=False)['score']
        results[i] = play_q_game(q_table, env, show_prints=False, show_gui=False)['score']

    end = datetime.now()

    # plt.subplot(1,3,1)
    # plt.title("h_dist")
    # plt.hist(observation_history[:,0], bins=40)
    # plt.subplot(1,3,2)
    # plt.title("v_dist")
    # plt.hist(observation_history[:,1], bins=40)
    # plt.subplots_adjust(wspace=0.3)

    if overwrite:
        fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.8)
        axs[0].plot(selected_scores, 'r', linewidth=0.7)
        axs[0].set_title("Learning progress")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Points scored")
        # axs[0].set_xlim([0,repeat])

        axs[1].plot(all_rewards, 'r', linewidth=0.7)
        axs[1].set_title("Rewards")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Rewards earned")
        # axs[1].set_xlim([0,repeat])

    if repeat:
        fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.8)

        axs[0].plot(results, 'r', linewidth=0.7)
        axs[0].set_title("Learning progress")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Points scored")

        axs[1].hist(results, bins=50, color='red')
        axs[1].set_ylabel("Frequency")
        axs[1].set_xlabel("Points scored")
        axs[1].set_title("Took %.2f mins"%((end-start).seconds/60))
        axs[1].set_xlim(0)

    plt.savefig("q_learning_results.png")


def q_learning(env, q_table):
    """Training the agent"""
    # For progress bar
    bar_length = 30

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    repeat = 100000

    # For plotting metrics
    all_scores = np.zeros(repeat)
    all_rewards = np.zeros(repeat)
    # observation_history = np.zeros((1,nr_feats))
    # states_visited = [[], []]

    for i in range(repeat):

        # For progress bar
        if (i+1) % (repeat/200) == 0:
            percent = 100.0*i/(repeat-1)
            sys.stdout.write('\r')
            sys.stdout.write("\rTraining progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()


        obs = env.reset()
        state = discrete_state(obs)
        done = False

        total_rewards = 0
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                # print(q_table[state])
                # print(np.argmax(q_table[state]))
                action = np.argmax(q_table[state]) # Exploit learned values


            # if obs[1] < -0.05:
            #     action = 1 #flap
            # else:
            #     action = 0 #idle

            next_obs, reward, done, info = env.step(action)
            next_state = discrete_state(next_obs)
            # states_visited[0].append(next_state[0])
            # states_visited[1].append(next_state[1])
            # observation_history = np.concatenate((observation_history, next_obs.reshape(1,-1)))

            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state][action] = new_value

            state = next_state
            obs = next_obs
            total_rewards += reward

        # states_visited[0] = list(set(states_visited[0]))
        # states_visited[1] = list(set(states_visited[1]))
        all_rewards[i] = total_rewards
        all_scores[i] = info['score']

    # print("\n",len(states_visited[0]),len(states_visited[1]))

    return q_table, all_scores, all_rewards#, observation_history[1:]


def discrete_state(obs):
    return h_state(obs[0]), v_state(obs[1])

def h_state(h_dist):
    # states 0-99
    min_value = 0.0
    max_value = 0.6

    if h_dist < min_value:
        return 0
    elif h_dist > max_value:
        return nr_states_h-1

    #first scale between 0 and 1 then distribute over the remaining nr of states
    return int((h_dist-min_value)/(max_value-min_value) * (nr_states_h-2))

def v_state(v_dist):
    # states 0-99
    min_value = -0.10
    max_value = 0.10
    # min_value = -0.50
    # max_value = 0.50

    if v_dist < -0.3:
        return 0
    elif v_dist < -0.2:
        return 1
    elif v_dist < -0.15:
        return 2
    elif v_dist < min_value:
        return 3
    if v_dist > 0.3:
        return nr_states_v-1
    elif v_dist > 0.2:
        return nr_states_v-2
    elif v_dist > 0.15:
        return nr_states_v-3
    elif v_dist > max_value:
        return nr_states_v-4
    if v_dist < min_value:
        return 0
    elif v_dist > max_value:
        return nr_states_v-1

    #first scale between 0 and 1 then distribute over the remaining nr of states
    return int((v_dist-min_value)/(max_value-min_value) * (nr_states_v-8)) + 1

def play_q_game(q_table, env=0, show_prints=True, show_gui=True, fps=100):

    if not env:
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
        state = discrete_state(obs)
        action = np.argmax(q_table[state])

        # Processing:
        obs, reward, done, info = env.step(action)

        if show_prints:
            # if prev_score != info['score']:
            now = datetime.now().second
            if prev_sec != now:
            # if reward > 0:
                prev_sec = now
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


def play_game(env=0, show_prints=False, show_gui=False, fps=100):

    if not env:
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