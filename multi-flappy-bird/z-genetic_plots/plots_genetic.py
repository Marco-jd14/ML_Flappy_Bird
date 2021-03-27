


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from datetime import datetime


data_files=["50pop-20hidden-30gen-best_score-GN", "50pop-5hidden-30gen-best_score-GN", "50pop-10hidden-30gen-best_score-GN", "50pop-15hidden-30gen-best_score-GN"]
data_files = ["50pop-10hidden-30gen-best_score-GN","70pop-10hidden-30gen-best_score-GN","90pop-10hidden-30gen-best_score-GN","110pop-10hidden-30gen-best_score-GN"]

plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=0.4) #0.8
max_overall = 0
for name in data_files:
    results = genfromtxt('%s.csv'%name, delimiter=',')

    # plt.subplot(2,1,1)

    plt.plot(results)
    plt.ylabel("Score")
    plt.xlabel("Generation")
    plt.title("Best score per generation")
    plt.grid()
    plt.xlim(0)
    plt.ylim(0, max_overall+5)

    goat_scores = np.zeros_like(results)

    max_score = 0
    for i in range(len(results)):
        if results[i] > max_score:
            max_score = results[i]
            if max_score > max_overall:
                max_overall = max_score
        goat_scores[i] = max_score

    # plt.subplot(2,1,2)

    # plt.plot(goat_scores)

plt.legend(data_files)
plt.ylabel("Score")
plt.xlabel("Generation")
plt.title("Best overall score so far")
plt.grid()
plt.xlim(0)
plt.ylim(0, max_overall+5)

plt.savefig("Genetic_algo.png")


