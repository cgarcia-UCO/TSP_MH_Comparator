import numpy as np
import random

def random_search(N, f):
    def generate_permutation(N):
        # generate a random permutation of the first N natural numbers
        permutation = np.random.permutation(N)

        # return the permutation
        return permutation

    current_permutation = generate_permutation(N)
    current_costs = [f(current_permutation)]

    while True:
        current_permutation = generate_permutation(N)
        current_costs.append(f(current_permutation))
