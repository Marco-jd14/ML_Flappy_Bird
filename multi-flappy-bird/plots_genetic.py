


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from datetime import datetime



results = genfromtxt('20node-50pop-50gen-best_score-GN.csv', delimiter=',')
print(results)

plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=0.4) #0.8
plt.subplot(2,1,1)

plt.plot(results,"k")
plt.ylabel("Score")
plt.xlabel("Generation")
plt.title("Best score per generation")
plt.grid()
plt.xlim(0)
plt.ylim(0)

goat_scores = np.zeros_like(results)

max_score = 0
for i in range(len(results)):
    if results[i] > max_score:
        max_score = results[i]
    goat_scores[i] = max_score

plt.subplot(2,1,2)

plt.plot(goat_scores,"k")
plt.ylabel("Score")
plt.xlabel("Generation")
plt.title("Best overall score so far")
plt.grid()
plt.xlim(0)
plt.ylim(0)

plt.savefig("Genetic_algo.png")


