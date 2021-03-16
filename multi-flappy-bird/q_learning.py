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

population_size = 100
q_its = int(100000/population_size)

exp_pop_size = 5                    #every bird does exactly the same thing anyway (based on q_table)
exp_its = int(2000/exp_pop_size)

# For progress bar
bar_length = 30


def main(options):
    env = flappy_bird_gym.make("FlappyBird-v0")

    train = 0
    test_result = 1
    gui_loops = 0

    filename = "q_table4 - Insane.npy"
    save_as = "q_table4.npy"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            q_table_before = np.load(f)
    else:
        q_table_before = np.zeros([nr_states_h * nr_states_v, env.action_space.n])

    # Uncomment if we want to start training from scratch:
    # q_table_before = np.zeros([nr_states_h * nr_states_v, env.action_space.n])

    if train:
        start1 = datetime.now()

        q_table, all_scores = q_learning(env, np.copy(q_table_before))

        end1 = datetime.now()
        print("\nTraining took %.2f mins\n"%((end1-start1).seconds/60))
        np.save(save_as, q_table)
        np.save("all_scores.npy", all_scores)

    else:
        q_table = q_table_before

    print("The q_table was updated:", not np.all(q_table==q_table_before))
    print("Proportion of q_table that is empty: %.2f%%\n" %(len(q_table[np.all(q_table==0.0,axis=1)])/nr_states_h/nr_states_v*100))

    # Uncomment to see the GUI play a game based on the q_table:
    for i in range(gui_loops):
        play_q_game(q_table, env, show_prints=False, fps=40, pop_size=1)

    if test_result:
        start2 = datetime.now()

        results = np.zeros(exp_its*exp_pop_size)
        for i in range(exp_its):
            results[i*exp_pop_size:(i+1)*exp_pop_size] = play_q_game(q_table, env, show_prints=False, show_gui=False)[:]

            # Progress bar
            if i % int(exp_its/100 + 1) == 0:
                percent = 100.0*i/(exp_its-1)
                sys.stdout.write('\r')
                sys.stdout.write("\rExperiment progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
                sys.stdout.flush()

        end2 = datetime.now()
        print("\nExperiment took %.2f mins\n"%((end2-start2).seconds/60))
        print("Fraction of games scored zero points: %.2f%%" %(len(results[results==0])/len(results)*100) )


    #PLOTTING THE RESULTS:

    if train:
        fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.8)

        axs[0].plot(all_scores, 'ob', alpha=0.1, markersize=2)
        axs[0].plot(np.arange(len(all_scores))[all_scores>=5], all_scores[all_scores>=5], 'ob', alpha=0.5, markersize=2)
        axs[0].set_title("Learning progress")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Points scored")
        axs[0].set_xlim([0,len(all_scores)])

        freq, bins, patches = axs[1].hist(all_scores, bins=50, color='red')
        axs[1].set_ylabel("Frequency")
        axs[1].set_xlabel("Points scored")
        axs[1].set_title("Took %.2f mins"%((end1-start1).seconds/60))
        axs[1].set_xlim(0)

        # For histogram bin labels
        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        for n, (fr, x, patch) in enumerate(zip(freq, bin_centers, patches)):
          height = int(freq[n])
          if height > 0:
              plt.annotate("%.1f%%"%(height/len(all_scores)*100),
                           size = 8,
                           xy = (x, height),             # top left corner of the histogram bar
                           xytext = (0,0.2),             # offsetting label position above its bar
                           textcoords = "offset points", # Offset (in points) from the *xy* value
                           ha = 'center', va = 'bottom'
                           )

        plt.savefig("q_learning_results.png")

    if test_result:
        fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.8)

        axs[0].plot(results, 'ob', alpha=0.1, markersize=2)
        axs[0].plot(np.arange(len(results))[results>=5], results[results>=5], 'ob', alpha=0.5, markersize=2)
        axs[0].set_title("After learning")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Points scored")
        axs[0].set_xlim([0,len(results)])

        freq, bins, patches = axs[1].hist(results, bins=50, color='red')
        axs[1].set_ylabel("Frequency")
        axs[1].set_xlabel("Points scored")
        axs[1].set_title("Took %.2f mins"%((end2-start2).seconds/60))
        axs[1].set_xlim(0)

        # For histogram bin labels
        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        for n, (fr, x, patch) in enumerate(zip(freq, bin_centers, patches)):
          height = int(freq[n])
          if height > 0:
              plt.annotate("%.1f%%"%(height/(exp_its*exp_pop_size)*100),
                           size = 8,
                           xy = (x, height),             # top left corner of the histogram bar
                           xytext = (0,0.2),             # offsetting label position above its bar
                           textcoords = "offset points", # Offset (in points) from the *xy* value
                           ha = 'center', va = 'bottom'
                           )

        plt.savefig("q_playing_results.png")



def q_learning(env, q_table):
    """Training the agent"""
    # Progress bar:
    sys.stdout.write("Training progress: [{:{}}] {:>3}%".format('='*int(0),bar_length, int(0)))
    sys.stdout.flush()

    # Hyperparameters
    alpha = 0.05        #0.1    #learning rate
    gamma = 0.95        #0.6    #discount rate
    epsilon = 0.05       #0.1

    # For plotting metrics
    all_scores = np.zeros(q_its*population_size)

    for i in range(q_its):
        obs = env.reset(population_size)
        states = discrete_state(obs)

        done = [False for k in range(population_size)]
        prev_done = np.copy(done)

        while not all(done):
            actions = np.argmax(q_table[states],axis=1)
            for j in range(population_size):
                if random.uniform(0, 1) < epsilon:
                    actions[j] = env.action_space.sample()  # Explore action space

            next_obs, rewards, done, scores = env.step(actions)
            next_states = discrete_state(next_obs)

            old_values = q_table[states,actions]
            next_maxs = np.max(q_table[next_states], axis=1)

            # print("\n")
            # print(q_table[states,actions])
            new_values = (1 - alpha) * old_values + alpha * (rewards + gamma * next_maxs)
            new_values[prev_done] = old_values[prev_done]  #only update in q_table for birds that are alive (or just died)
            q_table[states,actions] = new_values
            # print(states)
            # print(q_table[states,actions])

            prev_done = done
            states = next_states

        all_scores[i*population_size:(i+1)*population_size] = scores[:]

        # For progress bar
        if i % int(q_its/100+5) == 0:
            percent = 100.0*i/(q_its-1)
            sys.stdout.write('\r')
            sys.stdout.write("\rTraining progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()

    return q_table, all_scores


def play_q_game(q_table, env=0, show_prints=True, show_gui=True, fps=100, pop_size=exp_pop_size):

    if not env:
        env = flappy_bird_gym.make("FlappyBird-v0")
    obs = env.reset(pop_size)

    if show_gui:
        env.render()

    # q_table = np.zeros([nr_states_h * nr_states_v, env.action_space.n])

    epsilon = 0.0  #for trying
    prev_sec = -1
    while True:
        if show_gui:
            pygame.event.pump()

        states = discrete_state(obs)
        actions = np.argmax(q_table[states],axis=1)
        for j in range(population_size):
            if random.uniform(0, 1) < epsilon:
                actions[j] = env.action_space.sample()  # Explore action space

        obs, reward, done, scores = env.step(actions)

        if show_prints:
            if datetime.now().second != prev_sec:  # only print once per second
                prev_sec = datetime.now().second
                print("")
                for i in range(len(obs)):
                    print("BIRD %d:\t"%i, obs[i], "\tReward:", reward[i], "\tdied:",done[i], "\tinfo:",scores[i])

        # Rendering the game:
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

        if all(done):
            break

    env.close()

    return scores


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
    return int((v_dist-min_value)/(max_value-min_value) * (nr_states_v-8)) + 4




def play_game(env=0, show_prints=False, show_gui=False, fps=100):

    if not env:
        env = flappy_bird_gym.make("FlappyBird-v0")
    obs = env.reset()

    if show_gui:
        env.render(exp_pop_size)

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
        obs, reward, done, scores = env.step(action)

        if show_prints:
            now = datetime.now().second
            if prev_sec != now:
                prev_sec = now
                print("")
                for i in range(len(obs)):
                    print("BIRD %d:\t"%i, obs[i], "\tReward:", reward[i], "\tdied:",done[i], "\tinfo:",scores[i])

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