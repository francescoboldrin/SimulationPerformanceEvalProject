"""
Model implementation of the contagion process on a hypergraph provided by:

**Description of the contagion process:**
Mathematically, in the social contagion process the states of the nodes are modeled
as Bernoulli random variables, 𝑌𝑖 = 1 (with its complementary 𝑋𝑖 = 0) if the
node is active and 𝑌𝑖 = 0 otherwise (and then 𝑋𝑖 = 1). Individual states change
either spontaneously or as a consequence of their interactions. Formally, this is
a collection of independent Poisson processes. First, we associate to each active
node 𝑖 a Poisson process with parameter 𝛿𝑖
, modeling its spontaneous deactivation, {𝑌𝑖 = 1}𝛿𝑖−−→ {𝑋𝑖 = 1}. This transition is similar to the healing in disease spreading
dynamics. On the other hand, spreading processes occur along the hyperedges as
follows. For each hyperedge 𝑒 𝑗 we define a random variable 𝑇𝑗 =Í 𝑘 ∈𝑒𝑗 𝑌𝑘 : 𝑇𝑗 is by definition the number of active nodes in the hyperedge. If 𝑇𝑗ùis equal to or above a given threshold Θ𝑗
, we model the contagion by a Poisson process with parameter𝜆 𝑗. In other words, if 𝑇𝑗 ≥ Θ𝑗, then {𝑋𝑘 = 1}𝜆𝑗−−→ {𝑌𝑘 = 1}, ∀𝑘 ∈ 𝑒 𝑗.
This corresponds to a threshold process that becomes active only above a critical mass of active nodes. Finally, if |𝑒 𝑗| = 2, we assume directed Poisson processes, recoveringa traditional SIS contagion process.
For the sake of simplicity, we assume that 𝛿𝑖 = 𝛿and 𝜆 𝑗 = 𝜆 × 𝜆∗(|𝑒 𝑗|), where 𝜆 is the control parameter and 𝜆∗(|𝑒 𝑗|) is an arbitrary
function of the cardinality of the hyperedge.

**Implementation variations:**
We will consider only the variable state, associated with each node that refers to Yi (is the node infected) variable on the article.
Since Xis the complementary of Yi, for the sake of simplicity, we will not consider it in the implementation.

The Poisson process is implemented using the numpy library, which provides a function to generate random numbers from a Poisson distribution.
For each node in the hyperedge, we generate a random number from a Poisson distribution with mean 𝜆 × log2(|𝑒 𝑗|), where |𝑒 𝑗| is the cardinality of the hyperedge.
If the generated number is greater than 0, the node is infected, and its state is set to 1. (the event occurs at least once)

The threshold for sake of implementation is a ratio fixed.
For each hyperedge, we calculate the ratio of infected nodes to the cardinality of the hyperedge.
If the ratio is greater than or equal to the threshold, we spread the contagion through the hyperedge.

The choice of the seeds is made using the random strategy, where we randomly select a set of nodes to be the seeds.
We can also implement other strategies, but that is not the focus of this implementation.

A shuffle of the hyperedges is made before the contagion process starts to avoid bias in the contagion process.
In fact the contagion process designed in the article is higly susceptible to the order of the hyperedges.
"""

# define the class ContagionExecutive
from math import floor, ceil
from random import random
import numpy as np

import matplotlib.pyplot as plt


class ContagionExecutive:
    def __init__(self, hypergraph, seed_size, max_iter, threshold, seed_choice_strategy, contagion_lambda, recovery_mu):
        """
        -- ContagionExecutive --\n
        provide the contagion executive to run the contagion process on the hypergraph

        :param hypergraph: the hypergraph to run the contagion process on
        :param seed_size: the size of the seed set
        :param max_iter: the maximum number of iterations
        :param threshold: the threshold to spread the contagion
        :param seed_choice_strategy: the strategy to choose the seeds
        :param contagion_lambda: the contagion parameter
        :param recovery_delta: the recovery parameter

        initialize the contagion executive
        """
        self.hypergraph = hypergraph
        self.seed_size = seed_size
        self.max_iter = max_iter
        self.seed_choice_strategy = seed_choice_strategy

        self.threshold = threshold
        self.contagion_lambda = contagion_lambda
        self.recovery_mu = recovery_mu

    def run(self, is_antithetic):
        """
        run the contagion process on the hypergraph

        :return: the contagion function (the ratio of infected nodes at each iteration)

        **Contagion Algorithm:**
            - initialize the seeds with the algorithm chosen
            - initialize the contagion function as 0 for all the rounds
            - set the initial state of the nodes in the hypergraph to 0 (not infected) or 1 (infected) based on the seeds set
            - initialize the number of iterations as max iter
            - initialize the number of infected nodes to |seeds set| / |nodes set|
            - while the number of iterations is less than the maximum number of iterations:
                - update the contagion function
                - run the recovery process for each infected node
                - for each hyperedge:
                    - count the number of infected nodes in the edge
                    - if the number of infected nodes is greater than the threshold:
                        - for each node not infected in the edge:
                            - do contagion process with mean (lambda * lamba*(|size of the edge|)
                - update the number of infected nodes
        """
        # initialize seeds
        seeds = self.choice_seeds(is_antithetic)

        # define a contagion function of time to monitor the contagion process at each iteration, (index = iteration, value = ratio of infected nodes)
        contagion_function = []

        # set the initial state of the nodes in the hypergraph to 0 (not infected) or 1 (infected) based on the seeds
        for node in self.hypergraph.get_nodes():
            if node in seeds:
                # set the state of the node to 1 (infected)
                self.hypergraph.get_states()[node] = 1
            else:
                self.hypergraph.get_states()[node] = 0

        # initialize the number of iterations
        iter_counter = 0
        # initialize the number of infected nodes
        infected_nodes = len(seeds)

        while iter_counter < self.max_iter:
            # update the contagion function at each iteration
            outbreak = sum(self.hypergraph.get_states()) / self.hypergraph.get_number_of_nodes()
            contagion_function.append(outbreak)

            # set recovery process
            for node in self.hypergraph.get_nodes():
                if self.hypergraph.get_states()[node] == 1:
                    # recover the node with probability equals to success in poisson process with mean recovery_mu (delta_i in the paper)
                    if np.random.poisson(self.recovery_mu) > 0:
                        self.hypergraph.get_states()[node] = 0
                        infected_nodes -= 1

            # run the contagion process
            for hyperedge in self.hypergraph.get_hyperedges():
                infected_nodes += self.contagion_process(hyperedge)

            # check if the contagion process is over
            if infected_nodes == 0:
                # print("Contagion process is defeated at iteration: ", iter_counter)
                break

            # increment the number of iterations
            iter_counter += 1

        return contagion_function

    def contagion_process(self, hyperedge):
        # count the number of infected nodes in the hyperedge
        infected_nodes = sum([self.hypergraph.get_states()[node] for node in hyperedge.get_nodes()])
        if infected_nodes == 0:
            return 0
        ratio_infection = infected_nodes / hyperedge.get_size()

        new_infected = 0
        # check if the number overcome the threshold
        if ratio_infection >= self.threshold:
            # spread the contagion through the hyperedge
            mean = self.contagion_lambda * np.log2(hyperedge.get_size())
            for node in hyperedge.get_nodes():
                if self.hypergraph.get_states()[node] == 0:
                    # spread the contagion through the node every node is infected with probability equals to success
                    # in poisson process with mean lambda * log2(len(hyperedge.get_nodes()))
                    # the funciton log2 is proposed in the paper to normalize and limit the size of the hyperedge
                    if np.random.poisson(mean) >= 1:
                        self.hypergraph.get_states()[node] = 1
                        infected_nodes += 1
                        new_infected += 1
        return new_infected

    def choice_seeds(self, flag_antithetic):
        if self.seed_choice_strategy == "random":
            if flag_antithetic:
                return self.antithetic_seeds()
            return self.random_seeds()
        elif self.seed_choice_strategy == "degree":
            # return self.degree_seeds()
            # TODO: Implement the degree seed choice strategy
            raise NotImplementedError("The degree seed choice strategy is not implemented yet")
        else:
            raise ValueError("The seed choice strategy is not supported")

    def random_seeds(self):
        size_seeds = floor(self.seed_size * self.hypergraph.get_number_of_nodes())
        return np.random.choice(self.hypergraph.get_nodes(), size_seeds, replace=False)

    def antithetic_seeds(self):
        size_seeds = int(self.hypergraph.get_number_of_nodes() - (self.seed_size * self.hypergraph.get_number_of_nodes()))
        return np.random.choice(self.hypergraph.get_nodes(), size_seeds, replace=False)

    def degree_seeds(self):
        # TODO: revise the degree seed choice strategy
        degrees = self.hypergraph.get_degrees()
        return np.argsort(degrees)[-self.seed_size:]

    def get_hypergraph(self):
        return self.hypergraph

    def plot_results(self, contagion_function, title="Contagion process"):
        """

        :param contagion_function:
        :param title:
        :return:
        """
        # take the number of iterations
        iterations = range(0, len(contagion_function), 1)
        plt.clf()

        # plot the contagion function
        plt.plot(iterations, contagion_function, color='blue',  linestyle='solid')

        # now exclude the first interaction to have a better visualization
        limit_min = int(0.15 * len(contagion_function))
        contagion_function = contagion_function[limit_min:]

        # show the maximum, the minimum and the average of the contagion function
        plt.axhline(y=max(contagion_function), color='r', linestyle='--')
        plt.axhline(y=min(contagion_function), color='g', linestyle='--')
        plt.axhline(y=sum(contagion_function) / len(contagion_function), color='y', linestyle='--')


        plt.legend(["Contagion function", f"Max = {max(contagion_function)}", f"Min = {min(contagion_function)}", f"Avg = {np.mean(contagion_function)}"])

        # make the plot more readable with grid and x and y ticks
        plt.xticks([i for i in range(0, len(contagion_function)+20, 5)])
        plt.yticks([i / 20 for i in range(0, 22)])
        plt.grid()

        plt.xlabel("Iterations")
        plt.ylabel("Ratio of infected nodes")
        plt.title("Contagion process")
        if title == "antithetic seed":
            plt.title("Contagion process with antithetic seed")
        plt.show()

    # def calculate_expected_outbreak(self):
    #    """
    #     d E[state[node]] / dt = -recovery_mu * state[node] + lambda * ( 1 - state[node] ) * sigma (all hyperedge i belong) * sigma (all k that overcome threshold from threshold per size to size of hyperedge)
    #     * log2(size_edge) * Probability (edge as k infected)
    #    """
    #
    #    for node in self.hypergraph.get_nodes():
    #         yi = self.hypergraph.get_states()[node]
    #         # calculate the derivative of the state of the node
    #         derivative = -self.recovery_mu * yi # negative part, refers to the recovery process
    #         positive_part = self.contagion_lambda * (1 - yi) # positive part, refers to the contagion process
    #         summatory = 0
    #         for hyperedge in self.hypergraph.get_hyperedges():
    #             if node in hyperedge.get_nodes():
    #                 k_start = ceil(self.threshold * hyperedge.get_size())
    #                 for k in range(k_start, hyperedge.get_size()):
    #                     summatory += np.log2(hyperedge.get_size()) * self.probability_edge_infected(hyperedge, k)

    # def probability_edge_infected(self, hyperedge, k):
    #     # calculate the probability that a hyperedge has exactly k infected nodes
    #     """
    #     :param hyperedge:
    #     :param k:
    #     :return:
    #
    #     the formula is given in the article, that is extracted through a series of mathematics process and through dicrete Fourier Transform
    #     at the end of the day
    #     P(K=k) = (1/(n+1))*sigma (for all l from 0 to size of the edge) C^l
    #     """
    #
    #     n = len(hyperedge)
    #     C = np.exp(2j * np.pi / (n + 1))
    #     pe_j_k = 0
    #     for l in range(n + 1):
    #         product = 1
    #         for m in range(n):
    #             product *= (1 + (C ** l - 1) * self.hypergraph.get_states()[m])
    #         pe_j_k += C ** (-l * k) * product
    #     return pe_j_k.real / (n + 1)


