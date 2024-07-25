from HyperGraph import HyperGraph
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw

def ParseInput(FileReader):
    # read the size of the hypergraph
    number_of_nodes = int(FileReader.readline())

    # read the number of hyperedges
    number_of_hyperedges = int(FileReader.readline())

    # read the hyperedges
    hyperedges = []

    for i in range(number_of_hyperedges):
        line = FileReader.readline()
        hyperedges.append(list(map(int, line.split())))

    # return the hypergraph
    return HyperGraph(number_of_nodes, number_of_hyperedges, hyperedges, [0 for i in range(number_of_nodes)])

def visualizingPowerlaw():
    param = 2.25
    max_value = 10
    min_value = 2

    x = np.linspace(min_value, max_value, 100)
    y = x ** -param
    plt.plot(x, y)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Power Law Distribution")
    plt.show()

    plt.clf()

    res = np.random.power(param, 1000)
    res = min_value + (max_value - min_value) * ( 1 - res )

    plt.hist(res, bins=100)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Power Law Distribution")
    plt.show()

    res = np.random.random(1000) ** (-1 / (param - 1))

    plt.hist(res, bins=100)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Power Law Distribution")
    plt.show()

    res = powerlaw.rvs(-param, size=1000)

    plt.hist(res, bins=100)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Power Law Distribution")
    plt.show()