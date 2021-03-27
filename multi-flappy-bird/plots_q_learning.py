


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from datetime import datetime

settings = "-1mil;0.99"
filename = "q_tables/all_scores%s.npy"%settings
with open(filename, 'rb') as f:
    results = np.load(f)

# results = genfromtxt('50-100-best_score-GN.csv', delimiter=',')
print(results)

plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=0.4) #0.8
plt.subplot(1,1,1)

# plt.plot(results,"o-k",alpha=0.5)
# plt.ylabel("Score")
# plt.xlabel("Generation")
# plt.title("Best score per generation")
# plt.grid()
# plt.xlim(0)
# plt.ylim(0)
print("average:", np.average(results))
print("std dev:", np.std(results))

goat_scores = np.zeros_like(results)

max_score = 0
for i in range(len(results)):
    if results[i] > max_score:
        max_score = results[i]
    goat_scores[i] = max_score

# plt.subplot(2,1,2)

plt.plot(goat_scores,"k")
plt.ylabel("Score")
plt.xlabel("Game")
plt.title("Best overall score so far - Q-Learning")
plt.grid()
plt.xlim(0)
plt.ylim(0)

plt.savefig("Genetic_algo.png")


