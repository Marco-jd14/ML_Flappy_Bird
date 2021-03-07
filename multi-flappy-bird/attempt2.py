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
population_size = 500
q_its = int(1000000/population_size)
exp_pop_size = 2

# For progress bar
bar_length = 30


def main(options):
    env = flappy_bird_gym.make("FlappyBird-v0")

    repeat = 1
    overwrite = 1
    filename = "q_table.npy"
    with open(filename, 'rb') as f:
        q_table_before = np.load(f)

    if overwrite or not os.path.isfile(filename):
        start1 = datetime.now()

        # q_table_before = np.zeros([nr_states_h * nr_states_v, env.action_space.n])
        # q_table, all_scores, all_rewards = q_learning(env, q_table_before)
        q_table, all_scores = q_learning(env, np.copy(q_table_before))

        np.save(filename, q_table)
        end1 = datetime.now()
        print("\nTraining took %.2f mins\n"%((end1-start1).seconds/60))

    # if overwrite:
    #     dec = 100
    #     selected_scores = np.zeros(int(len(all_scores)/dec))
    #     for i in range(len(selected_scores)):
    #         selected_scores[i] = np.average(all_scores[i*dec:(i+1)*dec])

    # with open(filename, 'rb') as f:
    #     q_table2 = np.load(f)

    print(np.all(q_table==q_table_before))

    # play_q_game(q_table, env, fps=20)
    print("Proportion of q_table that is empty: %.2f%%\n" %(len(q_table[np.all(q_table==0.0,axis=1)])/nr_states_h/nr_states_v*100))

    repeat = int(20000/exp_pop_size)
    start2 = datetime.now()
    results = np.zeros(repeat*exp_pop_size)
    for i in range(repeat):
        if (i+1) % int(repeat/200+1) == 0:
            percent = 100.0*i/(repeat-1)
            sys.stdout.write('\r')
            sys.stdout.write("\rExperiment progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()

        # results[i] = MarcoCarlo.play_game(env, show_prints=False, show_gui=False)['score']
        results[i*exp_pop_size:(i+1)*exp_pop_size] = play_q_game(q_table, env, show_prints=False, show_gui=False)[:]

    end2 = datetime.now()
    print("\nExperiment took %.2f mins\n"%((end2-start2).seconds/60))


    if overwrite:
        fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.8)

        axs[0].plot(all_scores, 'ob', alpha=0.1, markersize=2)
        axs[0].plot(np.arange(len(all_scores))[all_scores>3], all_scores[all_scores>3], 'ob', alpha=0.8, markersize=2)
        axs[0].set_title("Learning progress")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Points scored")
        axs[0].set_xlim([0,len(all_scores)])

        axs[1].hist(all_scores, bins=50, color='red')
        axs[1].set_ylabel("Frequency")
        axs[1].set_xlabel("Points scored")
        axs[1].set_title("Took %.2f mins"%((end1-start1).seconds/60))
        axs[1].set_xlim(0)
        plt.savefig("q_learning_results.png")

    if repeat:
        fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.8)

        axs[0].plot(results, 'ob', alpha=0.1, markersize=2)
        axs[0].plot(np.arange(len(results))[results>3], results[results>3], 'ob', alpha=0.8, markersize=2)
        axs[0].set_title("After learning")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Points scored")

        axs[1].hist(results, bins=50, color='red')
        axs[1].set_ylabel("Frequency")
        axs[1].set_xlabel("Points scored")
        axs[1].set_title("Took %.2f mins"%((end2-start2).seconds/60))
        axs[1].set_xlim(0)
        plt.savefig("q_playing_results.png")



def q_learning(env, q_table):
    """Training the agent"""
    # q_table_start = np.copy(q_table)

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1


    # For plotting metrics
    all_scores = np.zeros(q_its*population_size)
    # all_rewards = np.zeros(q_its)
    # states_visited = [[], []]

    actions = np.zeros(population_size)
    for i in range(q_its):

        # For progress bar
        if (i+1) % (q_its/200) == 0:
            percent = 100.0*i/(q_its-1)
            sys.stdout.write('\r')
            sys.stdout.write("\rTraining progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()


        obs = env.reset(population_size)
        states = discrete_state(obs)

        # actions = np.argmax(q_table[states],axis=1)
        done = [False for k in range(population_size)]

        total_rewards = 0
        prev_done = np.copy(done)
        # max_score = 0
        while not all(done):
            actions = np.argmax(q_table[states],axis=1)
            for j in range(population_size):
                if random.uniform(0, 1) < epsilon:
                    actions[j] = not actions[j] #env.action_space.sample() # Explore action space

            next_obs, rewards, done, scores = env.step(actions)
            next_states = discrete_state(next_obs)

            old_values = q_table[states,actions]
            next_maxs = np.max(q_table[next_states],axis=1)

            new_values = (1 - alpha) * old_values + alpha * (rewards + gamma * next_maxs)
            new_values[prev_done] = old_values[prev_done]

            q_table[states,actions] = new_values

            prev_done = done
            states = next_states
            obs = next_obs
            # max_score = max(max_score,np.max(scores))
            # total_rewards += rewards


        # all_rewards[i] = total_rewards
        all_scores[i*population_size:(i+1)*population_size] = scores[:]

        # print(np.all(q_table==q_table_start))

    return q_table, all_scores#, all_rewards


def discrete_state(obs):
    states = np.zeros(population_size, dtype=int)
    for bird in range(len(obs)):
        states[bird] = h_state(obs[bird][0])*nr_states_v + v_state(obs[bird][1])

    return states

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

    #first scale between 0 and 1 then distribute over the remaining nr of states
    return int((v_dist-min_value)/(max_value-min_value) * (nr_states_v-8)) + 1


def play_q_game(q_table, env=0, show_prints=True, show_gui=True, fps=100):

    if not env:
        env = flappy_bird_gym.make("FlappyBird-v0")
    obs = env.reset(exp_pop_size)

    if show_gui:
        env.render()

    prev_score = -1
    prev_sec = -1
    while True:
        if show_gui:
            pygame.event.pump()

        obs = env._get_observation()
        states = discrete_state(obs)
        actions = np.argmax(q_table[states],axis=1)

        # Processing:
        obs, reward, done, scores = env.step(actions)

        if show_prints:
            if datetime.now().second != prev_sec:
                prev_sec = datetime.now().second
                print("")
                for i in range(len(obs)):
                    print("BIRD %d:\t"%i, obs[i], "\tReward:", reward[i], "\tdied:",done[i], "\tinfo:",info[i])

        # Rendering the game:
        # (remove this two lines during training)
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

        # Checking if the player is still alive
        if all(done):
            break

    env.close()

    return scores


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