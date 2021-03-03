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

    observation_history = np.zeros((1,nr_feats))
    for i in range(100):
        info, obs = play_game(options.verbose, options.show_gui, options.fps)
        observation_history = np.concatenate((observation_history, obs))

    observation_history = observation_history[1:]

    print("Minima:", np.min(observation_history,axis=0))
    print("Maxima:", np.max(observation_history,axis=0))

    # Minima: [  0.00347222  -0.51070313     -8.        ]
    # Maxima: [  1.64236111   0.52148438    380.48      ]

    plt.subplot(1,3,1)
    plt.title("h_dist")
    plt.hist(observation_history[:,0], bins=40)
    plt.subplot(1,3,2)
    plt.title("v_dist")
    plt.hist(observation_history[:,1], bins=40)
    plt.subplots_adjust(wspace=0.3)
    # plt.savefig("hist.png")
    # plt.subplot(1,3,3)
    # plt.hist(observation_history[:,2])

    h_range = np.linspace(0.0, 1.65, 1000)
    v_range = np.linspace(-0.52, 0.53, 1000)

    h_states = np.zeros_like(h_range)
    v_states = np.zeros_like(v_range)

    for i in range(len(h_range)):
        h_states[i] = h_state(h_range[i])
    for i in range(len(v_range)):
        v_states[i] = v_state(v_range[i])

    # print(h_states)
    # print(v_states)
    print(len(set(h_states)))
    print(len(set(v_states)))


    # env = flappy_bird_gym.make("FlappyBird-v0")
    # q_table = np.zeros([env.observation_space.n, env.action_space.n])

def h_state(h_dist):
    assert(h_dist>=0)
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

    observation_history = np.zeros((1,nr_feats))

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
        # action = 0

        # Processing:
        obs, reward, done, info = env.step(action)
        observation_history = np.concatenate((observation_history, obs.reshape(1,-1)))

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

    return info, observation_history[1:]



def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):

        Action_probabilities = np.ones(num_actions,
                dtype = float) * epsilon / num_actions

        print(state)
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction




def qLearning(env, num_episodes, discount_factor = 1.0,
                            alpha = 0.6, epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats






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