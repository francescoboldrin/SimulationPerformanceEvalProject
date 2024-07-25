# hypergraph generator

import numpy as np
import random
from Nodes import Node
from HyperEdge import Hyperedge
from HyperGraph import HyperGraph

global DEBUG
DEBUG = False


def contains_hyperedge(hyperedges, hyperedge):
    for h in hyperedges:
        if h.get_nodes() == hyperedge.get_nodes():
            return True
    return False

def remove_hyperedge(hyperedges, hyperedge):
    hyperedges.remove(hyperedge)


def check_redudancy(hyperedges):
    print("checking redundancy")
    print (len(hyperedges))
    length = len(hyperedges)
    # for i in range(length - 1):
    #     for j in range(i + 1, length):
    #         try:
    #             if hyperedges[i].get_nodes() == hyperedges[j].get_nodes():
    #                 print("redundant hyperedge in position: ", i, " and ", j)
    #                 hyperedges.remove(hyperedges[j])
    #                 length -= 1
    #         except IndexError:
    #             print("Index out of range")
    #             print (i, j)
    #             continue
    i = 0
    while i < length - 1:
        j = i + 1
        while j < length:
            if hyperedges[i].get_nodes() == hyperedges[j].get_nodes():
                print("redundant hyperedge in position: ", i, " and ", j)
                hyperedges.remove(hyperedges[j])
                length -= 1
            else:
                j += 1
        i += 1


def powerlaw_rvs(a, size=1):
    """Generate random variables from a power-law distribution.

    Parameters:
    a (float): The exponent parameter of the power-law distribution.
    size (int): The number of random variables to generate.

    Returns:
    numpy.ndarray: Random variables drawn from the power-law distribution.
    """
    size = (np.random.random(size) ** (-1 / (a - 1)))[0]

    if size < 2:
        size = 2

    if size > 10:
        size = 10

    return size


def powerlaw_shifted_rvs(a, x_min=1, size=1):
    """Generate random variables from a shifted power-law distribution.

    Parameters:
    a (float): The exponent parameter of the power-law distribution.
    x_min (float): The minimum value of the distribution.
    size (int): The number of random variables to generate.

    Returns:
    numpy.ndarray: Random variables drawn from the shifted power-law distribution.
    """
    u = np.random.random(size)
    value = x_min + (1 - u[0]) ** (-1 / (a - 1))

    if value < x_min:
        value = x_min

    if value > 19:
        value = 19

    return value






class HyperGraphGenerator:
    @staticmethod
    def generate_hypergraph(number_of_nodes, number_of_hyperedges, max_hyperedge_size, degree_nodes_distribution_param,
                            hyperedge_size_distribution_param):
        """
        Generate a hypergraph with the specified parameters
        :param number_of_nodes:
        :param number_of_hyperedges:
        :param max_hyperedge_size:
        :param degree_nodes_distribution_param: type of distribution and parameters
        :param hyperedge_size_distribution_param: type of distribution and parameters
        :return: hypergraph

        **Algorithm**
        1. Generate a list of nodes
        2. Initialize the degree of the nodes
        3. Generate a list of hyperedges
        4. Initialize the degrees of the nodes
        5. Choose a random hyperedge size
        6. Choose a random node
        7. Add the node to the hyperedge
        8. Decrement the degree of the node
        9. Add the hyperedge to the hyperedges list
        10. Repeat steps 5 to 9 until all nodes have degree 0
        11. Reset the degrees of the nodes
        12. Reset the sizes of the hyperedges
        13. Return the hypergraph
        """
        if number_of_nodes <= 0:
            raise ValueError("The number of nodes must be greater than 0")
        if number_of_hyperedges <= 0:
            raise ValueError("The number of hyperedges must be greater than 0")

        hyperedges = []

        nodes = [Node(i) for i in range(number_of_nodes)]

        # initialize the degrees of the nodes
        for node in nodes:
            if degree_nodes_distribution_param[0] == "uniform":
                degree = random.randint(degree_nodes_distribution_param[1], degree_nodes_distribution_param[2])
            elif degree_nodes_distribution_param[0] == "poisson":
                degree = np.random.poisson(degree_nodes_distribution_param[1]) + 1
            elif degree_nodes_distribution_param[0] == "exponential":
                degree = np.random.exponential(degree_nodes_distribution_param[1])
            elif degree_nodes_distribution_param[0] == "powerlaw":
                degree = np.random.power(degree_nodes_distribution_param[1])# not correct
            else:
                raise ValueError("The degree distribution is not supported")

            node.set_degree(degree)

        # if DEBUG:
        #     for n in nodes:
        #         print(n.get_degree())


        degrees = [nodes[i].get_degree() for i in range(number_of_nodes)]

        debug_degrees = [n.get_degree() for n in nodes]

        if DEBUG:
            print("degrees: ", degrees)

        stop_condition = False
        tmp_nodes = []  # mask the node with at least one degree
        for node in nodes:
            if node.get_degree() > 0:
                tmp_nodes.append(node)

        while not stop_condition:
            if hyperedge_size_distribution_param[0] == "uniform":
                hyperedge_size = random.randint(hyperedge_size_distribution_param[1],
                                                hyperedge_size_distribution_param[2])
            elif hyperedge_size_distribution_param[0] == "poisson":
                hyperedge_size = np.random.poisson(hyperedge_size_distribution_param[1]) + 1
            elif hyperedge_size_distribution_param[0] == "exponential":
                hyperedge_size = np.random.exponential(hyperedge_size_distribution_param[1])
            elif hyperedge_size_distribution_param[0] == "powerlaw":
                hyperedge_size = powerlaw_shifted_rvs(hyperedge_size_distribution_param[1],1)
                # print(hyperedge_size)
            else:
                raise ValueError("The hyperedge size distribution is not supported")
                exit(1)

            hyperedge_size = int(hyperedge_size)
            hyperedge = Hyperedge(hyperedge_size)

            counter = 0
            # if DEBUG:
            #     print("chosing nodes for hyperedge")
            while len(hyperedge.get_nodes()) < hyperedge_size:
                if not tmp_nodes:
                    break
                node = random.choice(tmp_nodes)

                if degrees[node.get_id()] > 0 and not hyperedge.contains_node(node.get_id()):
                    hyperedge.add_node(node.id)
                    degrees[node.get_id()] -= 1
                    node.add_hyperedge(hyperedge)
                else:
                    counter += 1
                    if counter >= len(tmp_nodes) + 2:
                        stop_condition = True
                        break

                    if degrees[node.get_id()] < 1:
                        tmp_nodes.remove(node)
                        # if DEBUG:
                        #     print("node removed: ", node.get_id())

            # if DEBUG:
            #     print("hyperedge size: ", hyperedge_size, " hyperedge nodes: ", hyperedge.get_nodes())
            #     print("tmp nodes: ", tmp_nodes)
            hyperedges.append(hyperedge)
            # if DEBUG:
            #     print("hyperedge: ", hyperedge)
            #     print("degrees: ", degrees)

            if not tmp_nodes:
                stop_condition = True

        for node in nodes:
            node.reset_degree()

        for hyperedge in hyperedges:
            hyperedge.reset_size()

        if DEBUG:
            for hyperedge in hyperedges:
                print(hyperedge)

            # print nodes degrees as a list all in one line
            print("reset degrees: ", [n.get_degree() for n in nodes])
            print("original degrees: ", debug_degrees)

            # calculate the difference between the original degrees and the reset degrees using numpy
            print("difference: ", np.array(debug_degrees) - np.array([n.get_degree() for n in nodes]))


        # check_redudancy(hyperedges)

        return HyperGraph(number_of_nodes, len(hyperedges), hyperedges, [node for node in nodes], [0 for i in range(number_of_nodes)])


