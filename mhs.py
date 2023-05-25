import numpy as np

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

def r2(N,f):
    random_search(N,f)


def simulated_annealing(N, f):
    def two_opt(sol1, idx1, idx2):
        new_permutation = sol1.copy()
        min_idx = min(idx1, idx2)
        max_idx = max(idx1, idx2)
        new_permutation = np.array(
            list(sol1)[:min_idx] +
            list(sol1)[min_idx:max_idx][::-1] +
            list(sol1)[max_idx:])
        return new_permutation

    def generate_permutation(N):
        # generate a random permutation of the first N natural numbers
        permutation = np.random.permutation(N)

        # return the permutation
        return permutation

    T = 100
    alpha=0.9
    # set the initial state
    current_permutation = generate_permutation(N)
    best_permutation = current_permutation
    nonstop_iterations = 0

    # set the initial temperature
    current_temp = T

    # initialize the iteration counter
    i = 0

    # initialize the lists for storing costs and iterations
    current_costs = [f(current_permutation)]
    best_costs = [current_costs[-1]]
    current_cost = current_costs[-1]
    iterations = [0]

    # loop until the stopping criteria are met
    while True:
        # generate a new state by swapping two random points
        idx1, idx2 = np.random.choice(len(current_permutation), size=2,
                                      replace=False)
        new_permutation = two_opt(current_permutation, idx1, idx2)

        # calculate the cost of the new state
        # new_cost = tsp_instance.evaluate_and_resgister(new_permutation, alg_label)
        new_cost = f(
            current_permutation)

        # determine whether to accept the new state
        if new_cost < current_cost:
            current_permutation = new_permutation
            current_cost = new_cost
        elif current_temp > 0.0001:
            if np.exp((current_cost - new_cost) / current_temp) > np.random.rand():
                current_permutation = new_permutation
                current_cost = new_cost

        current_costs.append(current_cost)

        # update the best state if necessary
        if current_cost < best_costs[-1]:
            best_permutation = current_permutation
            best_costs.append(current_cost)
        else:
            best_costs.append(best_costs[-1])

        # decrease the temperature
        current_temp *= alpha

        # increase the iteration counter
        i += 1

        # add the current iteration to the list of iterations
        iterations.append(i)

    # return the best state and its cost
    return current_costs, best_costs
