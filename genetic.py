import time
import numpy as np
import pygame
import flappy_bird_gym
from argparse import ArgumentParser


# python is so ugly -_-b
# this class is not made for safety
# unexpected behaviour may occur if weights or inputs array is not complete
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CLEAR = '\u001b[0m'


class NNModel:
    # create a NNModel with one single hidden layer
    def __init__(self, hidden_weights, output_weights):
        # creating all hidden nodes
        self.hidden_nodes = []
        for weights in hidden_weights:
            self.hidden_nodes.append(self.Node(weights))
        print('{}hidden layer created:\n{}{}'.format(bcolors.OKGREEN, hidden_weights, bcolors.CLEAR))

        # creating output layer
        self.output = self.Node(output_weights)
        print('{}Output layer created: {}{}'.format(bcolors.OKGREEN, self.output.__str__(), bcolors.CLEAR))

    # activate all nodes and produce an output
    def predict(self, inputs):
        # the cute activation function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # get each node's dot product and activate them
        hidden_layer_result = []
        for node in self.hidden_nodes:
            node_result = sigmoid(np.dot(inputs, node.weights))
            hidden_layer_result.append(node_result)
            # deubg
            # print("Hidden: dot product = {} \t sigmoid = {}".format(np.dot(inputs, node.weights), node_result))
        hidden_layer_result = np.array(hidden_layer_result).T

        # get output node
        prediction = sigmoid(np.dot(hidden_layer_result, self.output.weights))
        print("Output: dot product = {} \t sigmoid = {}".format(np.dot(hidden_layer_result, self.output.weights),
                                                                prediction))
        return prediction

    # our perceptron without activation (each nodes in hidden layer and output node)
    class Node:
        def __init__(self, weights):
            self.weights = np.array([weight for weight in weights]).T
            # print('unactivated HiddenNode created: {}'.format(self.weights))

        def __str__(self):
            return self.weights


def start(show_prints=False, show_gui=False, fps=60, agents_num=10):
    # create gym
    env = flappy_bird_gym.make("FlappyBird-v0")
    # get initial observation
    obs = env.reset(agents_num)

    # if GUI enabled, initialise the window and clear events
    if show_gui:
        env.render()
        pygame.event.pump()

    # create NN models for each birds
    models = []
    for i in range(agents_num):
        # generate random weights
        hidden_weights = np.random.random((4, 2))
        output_weights = np.random.random(4)
        models.append(NNModel(hidden_weights, output_weights))

    input()
    # Training, record fitness, and play
    while True:
        # make decision
        actions = []
        for index in range(agents_num):
            actions.append(models[index].predict(obs[index]))

        # Processing:
        obs, reward, done, scores = env.step(actions)

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
                        default=10,
                        help="Specify in how many agents the game should has")

    options = parser.parse_args()

    start(show_prints=options.verbose, show_gui=options.show_gui, fps=options.fps, agents_num=options.agents_num)
