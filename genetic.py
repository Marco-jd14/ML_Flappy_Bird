import random
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
        # print('{}hidden layer created:\n{}{}'.format(bcolors.OKGREEN, hidden_weights, bcolors.CLEAR))

        # creating output layer
        self.output = self.Node(output_weights)
        # print('{}Output layer created: {}{}'.format(bcolors.OKGREEN, self.output.__str__(), bcolors.CLEAR))

        # record fitness
        self.fitness = 0
        # print("")

    def __lt__(self, other):
        return self.fitness > other.fitness

    # activate all nodes and produce an output
    def predict(self, inputs):
        # the cute activation function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # get each node's dot product and activate them
        hidden_layer_result = []
        for node in self.hidden_nodes:
            node_result = sigmoid(np.dot(node.weights, inputs))
            hidden_layer_result.append(node_result)
        hidden_layer_result = np.array(hidden_layer_result).T

        # get output node
        prediction = sigmoid(np.dot(hidden_layer_result, self.output.weights))
        # print("hidden layer results: ", hidden_layer_result, "\t output", prediction)
        return prediction

    # our perceptron without activation (each nodes in hidden layer and output node)
    class Node:
        def __init__(self, weights):
            self.weights = np.array([weight for weight in weights]).T
            # print('unactivated HiddenNode created: {}'.format(self.weights))

        def __str__(self):
            return self.weights


class Genetic:
    # generate a population with random models
    def __init__(self, agents_num, hidden_nodes_count, inputs_count):
        self.agents_num = agents_num
        self.hidden_nodes_count = hidden_nodes_count
        self.inputs_count = inputs_count
        self.models = []
        for _ in range(agents_num):
            # generate random weights
            hidden_weights = 2 * np.random.random((hidden_nodes_count, inputs_count)) - 1
            output_weights = 2 * np.random.random(hidden_nodes_count) - 1
            self.models.append(NNModel(hidden_weights, output_weights))

    def predict(self, all_inputs):
        results = []
        for i in range(self.agents_num):
            results.append(self.models[i].predict(all_inputs[i]))
        # print("all predictions", results)
        return results

    def sort_fitness(self):
        self.models.sort()

    def get_fitness(self):
        all_fitness = []
        for model in self.models:
            all_fitness.append(model.fitness)

        return all_fitness

    def evolve_population(self):
        # select winners and directly put them into new population
        winners = self.select(4)

        # offsprings
        offsprings = [self.crossover(winners[0], winners[1])]
        # 1 best two winners offspring

        # 3 random random winners' crossover
        for _ in range(3):
            parents = random.sample(winners, 2)
            offsprings.append(self.crossover(parents[0], parents[1]))

        # 2 direct copy of two winners
        offsprings.append(winners[np.random.randint(3)])
        offsprings.append(winners[np.random.randint(3)])

        # offspring mutation


        # store the latest population
        self.models.clear()
        for model in winners:
            self.models.append(model)
            # print("winner", model)

        for model in offsprings:
            self.models.append(model)
            # print("offspring", model)

        for model in self.models:
            model.fitness = 0

    def select(self, winner_count):
        # sort then gimme those best birds
        self.sort_fitness()
        return self.models[0:winner_count]

    def crossover(self, left: NNModel, right: NNModel) -> NNModel:
        # cross over out of random N neurons
        cut_point = np.random.randint(self.hidden_nodes_count)

        # exchange each other's neurons
        for i in range(cut_point):
            left_copy = left.hidden_nodes
            left.hidden_nodes[i] = right.hidden_nodes[i]
            right.hidden_nodes[i] = left_copy[i]

        # pick one of the result
        return left if np.random.randint(2) == 1 else right


def start(show_prints=False, show_gui=False, fps=60, agents_num=10):
    # create gym
    env = flappy_bird_gym.make("FlappyBird-v0")
    # get initial observation
    obs = env.reset(agents_num)

    # if GUI enabled, initialise the window and clear events
    if show_gui:
        env.render()
        pygame.event.pump()

    # create Genetic Algorithm Population with NN models
    birds = Genetic(agents_num, 6, 2)
    best_fit = 0

    # Training, record fitness, and play
    while True:
        # make decision
        actions = birds.predict(obs)

        # Processing:
        obs, reward, done, scores = env.step(actions)

        # record fitness
        for i in range(len(obs)):
            birds.models[i].fitness += reward[i]

        # if all agents died, start new population
        if all(done):
            # evaluate / sort by fitness
            # select top 4 as our winner
            # crossover
            # mutate
            # print("Birds fitness", birds.get_fitness())
            birds.sort_fitness()
            if birds.models[0].fitness > best_fit:
                best_fit = birds.models[0].fitness
                print(best_fit)

            birds.evolve_population()

            # new game
            obs = env.reset(agents_num)

        # logging
        if show_prints:
            for i in range(len(obs)):
                print("BIRD %d:\t" % i, obs[i], "\tReward:", reward[i], "\tdied:", done[i], "\tinfo", scores[i])
            print("\n")

        # Rendering the game:
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

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
