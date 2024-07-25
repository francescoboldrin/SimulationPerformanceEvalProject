"""
This is the main file of the project. It is used to run the contagion process on a hypergraph and to add noise to the hypergraph.

The project implement a proposed model for the contagion process on a hypergraph. The model is based on the article "Contagion in Heterogeneous Networks" by M. Boguna, R. Pastor-Satorras, and A. Vespignani.
The project furthermore perform some experiments on the model and on the hypergraph to ensure some properties of the model and to test the robustness of the model.

The main focus was to implement the model and to test it on a hypergraph. The model is based on the SIS model, but it is extended to a hypergraph.
we ensure the well-functioning of the model with toy examples.
Then we checked the correspondence of the model with the theoretical model proposed in the article.

For the purpose of the project we implemented a generator of syntheic hypergraphs, that can generate hypergraphs with different properties.
In future work, we can extend the project to use real-world hypergraphs and to test the model on them.
"""
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from ContagionExecutive import ContagionExecutive
from HyperEdge import Hyperedge
from HyperGraph import HyperGraph
from InputParser import ParseInput
from HyperGraphGenerator import HyperGraphGenerator
from NoiseGenerator import NoiseGenerator
from TheoricalSimulator import TheoricalSimulator

DEBUG = False


def load_hypergraph(FileReader):
    return ParseInput(FileReader)

def print_result(average_func, max_func, min_func):
    mean_average = np.mean(average_func)
    mean_max = np.mean(max_func)
    mean_min = np.mean(min_func)

    variance_average = np.var(average_func)
    variance_max = np.var(max_func)
    variance_min = np.var(min_func)

    # calculate the confidence interval
    confidence_interval_average = 1.96 * math.sqrt(variance_average) / math.sqrt(100)
    confidence_interval_max = 1.96 * math.sqrt(variance_max) / math.sqrt(100)
    confidence_interval_min = 1.96 * math.sqrt(variance_min) / math.sqrt(100)

    print("Average number of infected nodes: ", mean_average, " with variance: ", variance_average, " with confidence interval: ", confidence_interval_average)
    print("Maximum number of infected nodes: ", mean_max, " with variance: ", variance_max, " with confidence interval: ", confidence_interval_max)
    print("Minimum number of infected nodes: ", mean_min, " with variance: ", variance_min, " with confidence interval: ", confidence_interval_min)

# def Heteroneous_model_prediction(hypergraph, simulation_parameters):
#
#     # calculate k2 and k3
#     k2, k3 = hypergraph.get_degree_dist_bysize()
#
#     if DEBUG:
#         print("k2: ", k2)
#         print("k3: ", k3)
#
#     # calculate lambda_2 and lambda_3
#     lambda_ = simulation_parameters[4]
#     lambda_2 = lambda_ * 1
#     lambda_3 = lambda_ * np.log2(3)
#
#     # calculate beta_2 and beta_3
#     recovery_mu = simulation_parameters[5]
#     beta_2 = lambda_2 * k2 / recovery_mu
#     beta_3 = lambda_3 * k3 / recovery_mu
#
#     if DEBUG:
#         print("beta_2: ", beta_2)
#         print("beta_3: ", beta_3)
#         print("beta_c: ", 2 * np.sqrt(beta_3) - beta_3 )
#
#     # the prediciton of stabilized number of infected nodes rho is
#     rho_lower = (beta_3 - beta_2) - np.sqrt((beta_2 - beta_3) ** 2 - 4 * (1-beta_2) * beta_3)
#     rho_upper = (beta_3 - beta_2) + np.sqrt((beta_2 - beta_3) ** 2 - 4 * (1-beta_2) * beta_3)
#
#     rho_lower /= 2 * beta_3
#     rho_upper /= 2 * beta_3
#
#     return rho_lower, rho_upper




def plot_lambda_function(contagion_function):
    x = [c[0] for c in contagion_function]
    y = [c[1] for c in contagion_function]
    plt.clf()
    plt.plot(x, y, 'r', label='Average number of infected nodes')
    # make ticks
    plt.xticks(np.arange(0, 0.31, 0.01))
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.grid()
    plt.xlabel("Lambda")
    plt.ylabel("Average number of infected nodes")
    plt.title("Average number of infected nodes as a function of lambda")
    plt.show()

from mpl_toolkits.mplot3d import Axes3D


def plot_perc_function(outbreaks_function):
    # Convert the list to a numpy array for easier indexing
    data = np.array(outbreaks_function)
    perc_erased = data[:, 0]
    lambda_vals = data[:, 1]
    averages = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(perc_erased, lambda_vals, averages, c='r', marker='o')

    ax.set_xlabel('Percentage of Edges Erased')
    ax.set_ylabel('Lambda')
    ax.set_zlabel('Average Outbreak Size')

    plt.show()


# def stocastic_analysis_simulator_3d(simulation_parameters, toy_hypergraph):
#     """
#     This function is used to study the robustness of the contagion process on a hypergraph by erasing a percentage of edges and running the contagion process multiple times, and varying the lambda parameter.
#     :param simulation_parameters: [seed_size, max_iterations, threshold, seed_choice_strategy, contagion_lambda, recovery_mu]
#     :param toy_hypergraph: [HyperGraph]
#     :return: [void]
#
#     **Algorithm**
#     - Create a list of lambdas to test
#     - Create a list of percentage of edges to erase
#     - For each percentage of edges to erase
#         - Create a temporary hypergraph by erasing the percentage of edges
#         - For each lambda
#             - Run the contagion process
#             - Calculate the average number of infected nodes thorugh multiple runs and delete of the stabilization phase (first 5%)
#             - Store the result in a list [percentage of edges erased, lambda, average] (the discrete function of the behavior of the contagion process)
#     """
#     # make a function of lambda and percentage of edges erased
#     lambda_collection = [x / 100 for x in range(0, 101)]
#     perc_to_erase = 0
#     outbreaks_function = []
#     while perc_to_erase <= 1:
#         print("percentage of edges erased: ", perc_to_erase)
#         tmp_hypergraph = toy_hypergraph.copy()
#         tmp_hypergraph.erase_edges(perc_to_erase)
#
#         # # check if the original hypergraph is intact by plotting the number of edges
#         # print("original hypergraph")
#         # toy_hypergraph.display()
#         # print("erased hypergraph")
#         # tmp_hypergraph.display()
#         # print ("\n\n")
#         for lam in lambda_collection:
#             simulation_parameters[4] = lam
#             simulator = ContagionExecutive(tmp_hypergraph, *simulation_parameters)
#             # take a mean of 5 runs
#             average = 0
#             for j in range(10):
#                 contagion_result = simulator.run(False)
#                 contagion_result = contagion_result[int(0.05 * len(contagion_result)):]
#                 average += np.mean(contagion_result)
#             average /= 10
#             outbreaks_function.append([perc_to_erase, lam, average])
#
#         print("average: ", average)
#         perc_to_erase += 0.05
#
#     plot_perc_function(outbreaks_function)
#     pass

def function_lambda_analysis(toy_hypergraph, simulation_parameters):
    # take a lambda every 0.05
    lambda_collection = [x / 100 for x in range(0, 101)]
    lam = 0.

    contagion_function = [] # store the contagion function for each lambda, contagion_function[i] = [lambda_i, average]

    while lam <= 0.3:
        simulation_parameters[4] = lam
        simulator = ContagionExecutive(toy_hypergraph, *simulation_parameters)

        average = 0

        print(lam)

        for i in range(10):
            contagion_result = simulator.run(False)
            # exclude first 10% iterations
            contagion_result += contagion_result[int(0.05 * len(contagion_result)):]

            average += np.mean(contagion_result)

        average /= 10

        contagion_function.append([lam, average])
        lam += 0.01

    plot_lambda_function(contagion_function)

def plot_analyisis_results(outbreaks_function):
    # Outbreaks function is a 2d function of the type [perc_erased, avg_outbreak_size]
    data = np.array(outbreaks_function)
    x = [c[0] for c in data]
    y = [c[1] for c in data]

    plt.plot(x, y, 'r', label='Average number of infected nodes')
    # make ticks
    plt.xticks(np.arange(0, 1.1, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.xlabel('Percentage of Edges Erased')
    plt.ylabel('Average Outbreak Size')
    plt.title('Average Outbreak Size as a function of Percentage of Edges Erased')
    plt.show()

def analysis_robustness(simulation_parameters, toy_hypergraph):
    """
    This function is used to study the robustness of the contagion process on a hypergraph by erasing a percentage of edges and running the contagion process multiple times.
    :param simulation_parameters: [seed_size, max_iterations, threshold, seed_choice_strategy, contagion_lambda, recovery_mu]
    :param toy_hypergraph: [HyperGraph]
    :return: [void]

    **Algorithm**
    - Create a list of percentage of edges to erase
    - For each percentage of edges to erase
        - Create a temporary hypergraph by erasing the percentage of edges
        - Run the contagion process
        - Calculate the average number of infected nodes thorugh multiple runs and delete of the stabilization phase (first 5%)
        - Store the result in a list [percentage of edges erased, average] (the discrete function of the behavior of the contagion process)
    """
    print("----parameters----")
    print("seed_size: ", simulation_parameters[0])
    print("max_iterations: ", simulation_parameters[1])
    print("threshold: ", simulation_parameters[2])
    print("seed_choice_strategy: ", simulation_parameters[3])
    print("contagion_lambda: ", simulation_parameters[4])
    print("recovery_mu: ", simulation_parameters[5])
    toy_hypergraph.display()

    outbreaks_function = []
    perc_to_erase = 0
    while perc_to_erase <= 1:
        print("percentage of edges erased: ", perc_to_erase)
        tmp_hypergraph = toy_hypergraph.copy()
        tmp_hypergraph.erase_edges(perc_to_erase)

        simulator = ContagionExecutive(tmp_hypergraph, *simulation_parameters)
        # take a mean of 5 runs
        average = 0
        for j in range(20):
            contagion_result = simulator.run(False)
            contagion_result = contagion_result[int(0.05 * len(contagion_result)):]
            average += np.mean(contagion_result)
        average /= 20
        outbreaks_function.append([perc_to_erase, average])

        print("average: ", average)
        perc_to_erase += 0.05

    plot_analyisis_results(outbreaks_function)
    pass

def simple_run(simulation_parameters, toy_hypergraph):
    """
    This function is used to run the contagion process once and to plot the results.
    :param simulation_parameters:
    :param toy_hypergraph:
    :return:

    **Algorithm**
    - Run the contagion process
    - Plot
    """
    simulator = ContagionExecutive(toy_hypergraph, *simulation_parameters)
    toy_hypergraph.display()
    # toy_hypergraph.plot_edge_sizes_distribution("toy_hypergraph")
    # toy_hypergraph.plot_degree_distribution("toy_hypergraph")
    contagion_result = simulator.run(False)
    simulator.plot_results(contagion_result)


def multiple_runs(toy_hypergraph, simulation_parameters):
    """
    Monte Carlo simulation to calculate the average number of infected nodes over time.
    :param toy_hypergraph:
    :param simulation_parameters:
    :return:

    **Description**
    With the Monte Carlo methods combined with long runs, initialization cuts and antithetic runs, we can study the outbreaks of the contagion in a stable condition even if in a stochastic environment.
    The goal is to estimate the average number of infected nodes over time and to calculate the variance of the
    estimator of the average number of infected nodes.

    """
    toy_hypergraph.display()
    max_sim = 100
    outbreak_estimators = []
    outbreak_estimator_var = 0
    anti_flag = 1
    for i in range(max_sim):
        print(i)
        simulator = ContagionExecutive(toy_hypergraph, *simulation_parameters)
        contagion_result = simulator.run(False)
        if anti_flag == 1:
            cont_res_2 = simulator.run(True)
            contagion_result = [(contagion_result[i] + cont_res_2[i]) / 2 for i in range(len(contagion_result))]
        contagion_result = contagion_result[int(0.05 * len(contagion_result)):]
        outbreak_estimators.append(np.mean(contagion_result))

    outbreak_estimator_var = np.var(outbreak_estimators)

    # calculate CI for 0.95
    CI = 1.96 * math.sqrt(outbreak_estimator_var) / math.sqrt(max_sim)

    # round all to 4 decimal places
    CI = round(CI, 4)
    outbreak_estimator_var = round(outbreak_estimator_var, 4)
    outbreak_estimator = round(np.mean(outbreak_estimators), 4)

    # print the results
    print("Outbreak estimator: ", outbreak_estimator)
    print("Outbreak estimator variance: ", outbreak_estimator_var)
    print("Confidence interval: ", CI)


def plot_simulation_combined(theo_results, simul_results):
    # plot theo results as a red line
    x = [i for i in range(len(theo_results))]
    y = [theo_results[i] for i in range(len(theo_results))]

    plt.plot(x, y, 'r', label='Theorical results')

    # plot the simulation results as a blue line
    x = [i for i in range(len(simul_results))]
    y = [simul_results[i] for i in range(len(simul_results))]
    plt.plot(x, y, 'b', label='Simulation results')

    # add the legend
    plt.legend(['Theorical results', 'Simulation results'])

    # make ticks
    plt.xticks(np.arange(0, len(theo_results)+1, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Average number of infected nodes')
    plt.show()
def theorical_simulation(toy_hypergraph, simulation_parameters):
    """
    This function is used to compare the results of the contagion process with the theorical model proposed in the article.
    :param toy_hypergraph:
    :param simulation_parameters:
    :return:

    **Algorithm**
    - Calculate the function of the average number of infected nodes over time, with the theorical predictive model
    - Run the contagion process on the hypergraph
    - Calculate the average number of infected nodes over time
    - Plot the two functions on the same graph to compare the results
    """
    # theorical function prediction
    simulator = TheoricalSimulator(toy_hypergraph, simulation_parameters)
    results = simulator.main_ODE_function()

    # actual simulation
    simulation_parameters = [0.1, 50, 0.4, "random", 0.25, 0.5]  # [seed_size, max_iterations, threshold, seed_choice_strategy, contagion_lambda, recovery_mu]
    simulator_2 = ContagionExecutive(toy_hypergraph, *simulation_parameters)

    contagion_result = simulator_2.run(False)

    plot_simulation_combined(results, contagion_result)

    print("Theorical simulation done")


if __name__ == '__main__':
    toy_hypergraph = HyperGraphGenerator.generate_hypergraph(5000, 100, 5, ["poisson", 5], ["powerlaw", 2.25])

    simulation_parameters = [0.1, 100, 0.2, "random", 0.25, 2.]# [seed_size, max_iterations, threshold, seed_choice_strategy, contagion_lambda, recovery_mu]

    print("start simulation")

    """
    The following functions are used to test with various parameters the contagion process on a hypergraph.
    
    **Description of the functions**
    - function_lambda_analysis: This function is used to study the behavior of the contagion process as a function of lambda.
    - multiple_runs: This function is used to run the contagion process multiple times and to calculate the average number of infected nodes. (study the outbreak estimator)
    - simple_run: This function is used to run the contagion process once and to plot the results.
    - analysis_robustness: This function is used to study the robustness of the contagion process on a hypergraph by erasing a percentage of edges and running the contagion process multiple times.
    - theorical_simulation: This function is used to compare the results of the contagion process with the theorical model proposed in the article.
    """
    # function_lambda_analysis(toy_hypergraph, simulation_parameters)
    # multiple_runs(toy_hypergraph, simulation_parameters)
    simple_run(simulation_parameters, toy_hypergraph)
    # analysis_robustness(simulation_parameters, toy_hypergraph)
    # theorical_simulation(toy_hypergraph, simulation_parameters)




