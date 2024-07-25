# in this library it is contained the hyper graph class
# and the functions to create the hypergraph from the data

import numpy as np
import matplotlib.pyplot as plt
import random
import time


class HyperGraph:
    """
    HyperGraph class\n
    :param number_of_nodes: number of nodes in the hypergraph
    :param number_of_hyperedges: number of hyperedges in the hypergraph
    :param hyperedges: list of hyperedges, each hyperedge is an Hyperedge object (list of nodes + size)
    :param nodes: nodes of the hypergraph
    :param states: states of the nodes in the hypergraph (Susceptible, Infected)

    The class contains the hypergraph data structure and the functions to manipulate it
    """
    def __init__(self, number_of_nodes, number_of_hyperedges, hyperedges, nodes, states):
        """
        -- HyperGraph --\n
        :param number_of_nodes
        :param number_of_hyperedges
        :param hyperedges: list of hyperedges, each hyperedge is a list of nodes
        :param nodes: nodes of the hypergraph
        :param states: states of the nodes in the hypergraph (Susceptible, Infected)

        initialize the hypergraph
        """
        # check validity of the input
        if number_of_nodes <= 0:
            raise ValueError("The number of nodes must be greater than 0")
        if number_of_hyperedges <= 0:
            raise ValueError("The number of hyperedges must be greater than 0")
        if hyperedges is None or len(hyperedges) != number_of_hyperedges:
            print(len(hyperedges))
            print(number_of_hyperedges)
            raise ValueError("The hyperedges must be defined")
        if states is None or len(states) != number_of_nodes:
            raise ValueError("The states must be defined")
        if nodes is None or len(nodes) != number_of_nodes:
            raise ValueError("The nodes must be defined")

        # initialize the hypergraph
        self.number_of_nodes = number_of_nodes
        self.number_of_hyperedges = number_of_hyperedges
        self.hyperedges = hyperedges
        self.nodes = nodes
        self.states = states

    def __str__(self):
        return "HyperGraph: " + str(self.hyperedges)

    def shuffle_hyperedges(self):
        random.shuffle(self.hyperedges)

    def get_hyperedges(self):
        return self.hyperedges

    def reset_hyperedges(self, **kwargs):
        self.hyperedges = []
        if 'hyperedges' in kwargs:
            self.hyperedges = kwargs['hyperedges']

        if 'retainedges' in kwargs:
            self.hyperedges.extend(kwargs['retainedges'])

    def get_nodes(self):
        return [node.get_id() for node in self.nodes]

    def get_states(self):
        return self.states

    def get_number_of_nodes(self):
        return self.number_of_nodes

    def get_number_of_hyperedges(self):
        self.number_of_hyperedges = len(self.hyperedges)
        return self.number_of_hyperedges

    def reset_number_of_hyperedges(self):
        self.number_of_hyperedges = len(self.hyperedges)
    def get_average_edge_size(self):
        return sum([len(hyperedge.get_nodes()) for hyperedge in self.hyperedges]) / self.number_of_nodes

    def get_average_degree(self):
        degrees = [0 for i in range(self.number_of_nodes)]

        for hyperedge in self.hyperedges:
            for node in hyperedge.get_nodes():
                degrees[node] += 1

        return sum(degrees) / self.number_of_nodes

    def get_degrees(self):
        degrees = [0 for i in range(self.number_of_nodes)]

        for hyperedge in self.hyperedges:
            for node in hyperedge.get_nodes():
                degrees[node] += 1

        return degrees

    def get_degrees_distribution(self):
        degrees = self.get_degrees()
        degrees_distribution = [0 for i in range(0,100)]

        for degree in degrees:
            try:
                degrees_distribution[degree] += 1
            except IndexError:
                # extend the list
                degrees_distribution.extend([0 for i in range(degree - len(degrees_distribution) + 1)])
                degrees_distribution[degree] += 1

        # normalize the distribution
        degrees_distribution = [degree / self.number_of_nodes for degree in degrees_distribution]

        return degrees_distribution

    def get_edge_sizes(self):
        edge_sizes = [0 for i in range(0, 20)]

        for hyperedge in self.hyperedges:
            try:
                edge_sizes[len(hyperedge.get_nodes())] += 1
            except IndexError:
                # extend the list
                edge_sizes.extend([0 for i in range(len(hyperedge.get_nodes()) - len(edge_sizes) + 1)])
                edge_sizes[len(hyperedge.get_nodes())] += 1

        return edge_sizes

    def get_edge_sizes_distribution(self):
        edge_sizes = self.get_edge_sizes()

        print (edge_sizes)
        print ("\n\n\n")

        edge_sizes_distribution = [0 for i in range(0, 20)]

        sum_edge_sizes = sum(edge_sizes)
        for i in range(len(edge_sizes)):
            edge_sizes_distribution[i] = edge_sizes[i] / sum_edge_sizes

        return edge_sizes_distribution

    def plot_degree_distribution(self, title):
        plt.clf()
        degrees_distribution = self.get_degrees_distribution()
        plt.plot(degrees_distribution)
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        maximum = max(degrees_distribution)
        plt.xticks(range(0, len(degrees_distribution), 10))
        plt.yticks(np.arange(0, maximum + 0.01, round(maximum / 10, 2)))
        plt.grid()
        plt.title("Degree distribution")
        plt.show()


    def plot_edge_sizes_distribution(self, title):
        edge_sizes_distribution = self.get_edge_sizes_distribution()
        plt.clf()
        plt.plot(edge_sizes_distribution)
        plt.xlabel("Edge size")
        plt.ylabel("Frequency")
        plt.xticks(range(0, len(edge_sizes_distribution), 1))
        maximum = max(edge_sizes_distribution)
        plt.yticks(np.arange(0, maximum + 0.01, round(maximum / 10, 2)))
        plt.grid()
        plt.title("Edge size distribution")
        plt.show()




    def display(self):
        print("Number of nodes: ", self.number_of_nodes)
        print("Number of hyperedges: ", self.number_of_hyperedges)
        print("Actual number of hyperedges: ", len(self.hyperedges))
        print("Distribution of edge sizes: ", self.get_edge_sizes_distribution())
        print("Distribution of degrees: ", self.get_degrees_distribution())


    def get_degree_dist_bysize(self):
        # return the average degree of the nodes in the hypergraph for each size of the hyperedge
        MAX_HYPEREDGE_SIZE = 20
        degrees = []

        for i in range(self.get_number_of_nodes()):
            degrees.append([0, 0]) # a counter for size 2 and a counter for size 3

        for hyperedge in self.hyperedges:
            size = len(hyperedge.get_nodes())
            try:
                for node in hyperedge.get_nodes():
                    degrees[node][size - 2] += 1
            except IndexError:
                print("this calculation only works with hypergraph with hyperedges of size 2 and 3")
                return -1

        # calculate the average number of edge of size 2 are connected to a node
        k2 = 0
        for degree in degrees:
            k2 += degree[0]

        k2 /= self.get_number_of_nodes()

        # calculate the average number of edge of size 3 are connected to a node
        k3 = 0
        for degree in degrees:
            k3 += degree[1]

        k3 /= self.get_number_of_nodes()

        return k2, k3


    def erase_edges(self, percentage):
        # sample at random the edges to erase
        number_of_edges_to_erase = int(percentage * self.get_number_of_hyperedges())
        edges_to_erase = random.sample(self.hyperedges, number_of_edges_to_erase)

        # erase the edges
        for edge in edges_to_erase:
            self.hyperedges.remove(edge)

        self.reset_number_of_hyperedges()

    def copy(self):
        return HyperGraph(self.number_of_nodes, self.number_of_hyperedges, self.hyperedges.copy(), self.nodes.copy(), self.states.copy())





