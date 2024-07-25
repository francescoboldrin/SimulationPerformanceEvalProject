import numpy as np
import random
class NoiseGenerator:
    def __init__(self, noise_magnitude, noise_sign_toss_coin, percentage_edges_affected):
        self.noise_magnitude = noise_magnitude
        self.noise_sign_toss_coin = noise_sign_toss_coin
        self.percentage_edges_affected = percentage_edges_affected

    def add_noise(self, hypergraph):
        # add noise to the hypergraph
        hyperedges = random.sample(hypergraph.get_hyperedges(), int(self.percentage_edges_affected * hypergraph.get_number_of_hyperedges()))

        retainedges = []
        for hyperedge in hypergraph.get_hyperedges():
            if hyperedge not in hyperedges:
                retainedges.append(hyperedge)

        MAX_SIZE = 1000
        for hyperedge in hyperedges:
            MAX_SIZE = len(hyperedge.get_nodes()) * 2 - 1
            new_size = len(hyperedge.get_nodes()) + self.get_noise_amplitude(self.noise_magnitude, self.noise_sign_toss_coin, MAX_SIZE)

            if new_size < 2:
                hyperedges.remove(hyperedge)

            if new_size > MAX_SIZE:
                new_size = MAX_SIZE

            if new_size > len(hyperedge.get_nodes()):
                for i in range(new_size - len(hyperedge.get_nodes())):
                    counter = 0
                    while True:
                        node = random.choice(hypergraph.get_nodes())
                        counter += 1
                        if not hyperedge.contains_node(node):
                            hyperedge.add_node(node)

                            break
                        if counter > 1000:
                            break
            else:
                for i in range(len(hyperedge.get_nodes()) - new_size):
                    hyperedge.remove_node(random.choice(hyperedges.get_nodes()))

            for hyperedge in hyperedges:
                hyperedge.reset_size()

        hypergraph.reset_hyperedges(hyperedges=hyperedges, retainedges=retainedges)
        hypergraph.reset_number_of_hyperedges()

        return hypergraph


    def get_noise_amplitude(self, noise_magnitude, noise_sign_toss_coin, MAX_SIZE):
        # make an incremental binom random variable
        # stop is the success event and have probability 1 - noise_magnitude
        increment = 0
        while random.random() < noise_magnitude:
            increment += 1 # Probability of increment = k is P[increment = k] = (p)^(k-1) * (1-p), insuccess after k-1 successes
            if increment >= MAX_SIZE:
                break
        #
        # print ("Noise increment: ", increment)


        # toss a coin to decide the sign of the noise
        if random.random() > noise_sign_toss_coin:
            increment = -increment

        return increment