from math import ceil

import numpy as np
import random
import warnings


class TheoricalSimulator:
    def __init__(self, hypergraph, simulation_parameters):
        self.hypergraph = hypergraph
        self.simulation_parameters = simulation_parameters

    def compute_P_ej(self, K, n, y):
        P_ej_K = 0
        C = np.exp(2j * np.pi / (n + 1))

        for l in range(n + 1):
            C_l = C ** l
            product_term = 1
            for m in range(1, n + 1):
                product_term *= (1 + (C_l - 1) * y[m - 1])
            P_ej_K += (C_l ** -K) * product_term

        P_ej_K /= (n + 1)
        P_ej_K_real = np.real(P_ej_K)
        P_ej_K_clamped = np.clip(P_ej_K_real, 0, 1)

        return P_ej_K_clamped

    def update_y_i(self, i, y, delta, lambda_, hyperedges):
        """
        Update the state of the node i
        :param i:
        :param y:
        :param delta:
        :param lambda_:
        :param hyperedges:
        :return:

        **Algorithm**
        1. Initialize the dy_dt as -delta * y[i]
        2. Calculate the increment_counter
        3. For each hyperedge that contains the node i
            4. Get the size of the hyperedge
            5. Calculate the Theta_j
            6. Initialize the check_prob
            7. For each k in the range Theta_j to size_e_j + 1
                8. Calculate the P_ej_k
                9. Add the P_ej_k to the check_prob
                10. Calculate the lambda_star
                11. Increment the increment_counter by lambda_star * P_ej_k
        """
        try:
            dy_dt = -delta * y[i]
        except:
            print("Error: ", y, i, delta, lambda_, hyperedges)
            exit()
        increment_counter = 0
        for e_j in hyperedges:
            e_j = e_j.get_nodes()
            if i in e_j:
                size_e_j = len(e_j)
                Theta_j = ceil(self.simulation_parameters[2] * size_e_j)
                check_prob = 0
                for k in range(Theta_j, size_e_j + 1):
                    P_ej_k = self.compute_P_ej(k, size_e_j, [y[node] for node in e_j])
                    check_prob += P_ej_k
                    lambda_star = np.log2(size_e_j)
                    increment_counter += lambda_star * P_ej_k
                if check_prob > 1.01 or check_prob < 0:
                    print("Error: ", check_prob)
                    print("e_j: ", e_j)
                    print("size_e_j: ", size_e_j)
                    print("Theta_j: ", Theta_j)
                    print("y[nodes]",[y[node] for node in e_j])
                    exit()
        dy_dt += lambda_ * (1 - y[i]) * increment_counter

        return dy_dt

    def runge_kutta_step(self, y, delta, lambda_, hyperedges, dt):
        """
        Runge-Kutta step to update the states of the nodes
        :param y:
        :param delta:
        :param lambda_:
        :param hyperedges:
        :param dt:
        :return:

        **Algorithm**
        1. Initialize the k's
        2. Calculate k1 as the update of the state of the node i with y and dt
        3. Calculate k2 as the update of the state of the node i with y_temp = y + 0.5 * k1 * dt and dt
        4. Calculate k3 as the update of the state of the node i with y_temp = y + 0.5 * k2 * dt and dt
        5. Calculate k4 as the update of the state of the node i with y_temp = y + k3 * dt and dt
        6. Update each element of y as y[i] = y[i] + actual_dy where actual_dy = dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
        7. Return y
        """
        k1 = [0 for i in range(len(y))]
        k2 = [0 for i in range(len(y))]
        k3 = [0 for i in range(len(y))]
        k4 = [0 for i in range(len(y))]
        y_temp = [0 for i in range(len(y))]

        for i in range(len(y)):
            k1[i] =  self.update_y_i(i, y, delta, lambda_, hyperedges)
        # print("k1: ", k1)

        for i in range(len(y)):
            y_temp[i] = y[i] + 0.5 * k1[i] * dt
            if y_temp[i] < 0:
                y_temp[i] = 0
            elif y_temp[i] > 1:
                y_temp[i] = 1
        for i in range(len(y)):
            k2[i] = self.update_y_i(i, y_temp, delta, lambda_, hyperedges)
        # print("k2: ", k2)

        for i in range(len(y)):
            y_temp[i] = y[i] + 0.5 * k2[i] * dt
            if y_temp[i] < 0:
                y_temp[i] = 0
            elif y_temp[i] > 1:
                y_temp[i] = 1
        for i in range(len(y)):
            k3[i] = self.update_y_i(i, y_temp, delta, lambda_, hyperedges)
        # print("k3: ", k3)

        for i in range(len(y)):
            y_temp[i] = y[i] + k3[i] * dt
            if y_temp[i] < 0:
                y_temp[i] = 0
            elif y_temp[i] > 1:
                y_temp[i] = 1
        for i in range(len(y)):
            k4[i] =  self.update_y_i(i, y_temp, delta, lambda_, hyperedges)
        # print("k4: ", k4)

        # print("k1: ", k1
        #       , "\nk2: ", k2
        #       , "\nk3: ", k3
        #       , "\nk4: ", k4)

        # update each element
        for i in range(len(y)):
            actual_dy = dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
            y[i] = y[i] + actual_dy
            if y[i] < 0:
                y[i] = 0
            elif y[i] > 1:
                y[i] = 1

        return y

    def main_ODE_function(self):
        """
        Main function that integrates the ODE equation
        :param hypergraph:
        :param simulation_parameters:


        **Algorithm**
        1. Initialize the function_integrated
        2. Initialize the time
        3. Initialize the y array to the initial state (seed set of random nodes to 1 and the rest to 0)
        4. Initialize the function_integrated with the mean of the states at the beginning
        5. While the time is less than the final time
            6. Update the states using the runge_kutta_step function:
                for each node state:
                    7. Calculate k1
                    8. Calculate k2
                    9. Calculate k3
                    10. Calculate k4
                    11. Update the state with the weighted mean of the k's
            12. Append the mean of the states to the function_integrated
            13. Increment the time
        14. Return the function_integrated
        :return:
        """
        function_integrated = []  # function[index = time] = mean of the states
        time_f = 50
        time_start = 0
        time = time_start

        delta_ = self.simulation_parameters[5]
        lambda_ = self.simulation_parameters[4]
        theta_ = self.simulation_parameters[2]
        seed_size = self.simulation_parameters[0]

        y = [0 for i in range(self.hypergraph.get_number_of_nodes())]

        seed = np.random.choice(self.hypergraph.get_nodes(), int(seed_size * self.hypergraph.get_number_of_nodes()), replace=False)

        for node in seed:
            y[node] = 1

        hyperedges = self.hypergraph.get_hyperedges()

        dt = 1

        print("---parameters---")
        print("delta: ", delta_)
        print("lambda: ", lambda_)
        print("theta: ", theta_)
        print("seed size: ", seed_size)
        print("y: ", y)
        print("dt: ", dt)

        function_integrated.append(np.mean(y))
        while time < time_f:
            print("\n\n", y)
            y = self.runge_kutta_step(y, delta_, lambda_, hyperedges, dt)
            function_integrated.append(np.mean(y))

            time += 1

        return function_integrated

    # def P_ej_K_k(self, k, n, states, i, hyperedge):
    #     """
    #     Calculate the probability of a hyperedge of size j to have k nodes infected
    #     :param n: number of nodes in the hyperedge
    #     """
    #     # calculate the probability of a hyperedge of size j to have k nodes infected
    #     # Compute C_l = exp(2 * pi * i / (n + 1))
    #     C = np.exp((2j * np.pi) / (n + 1))
    #
    #     # if real part is close to 0, set it to 0
    #     if np.abs(np.real(C)) < 1e-10:
    #         C = 1j * np.imag(C)
    #
    #     # if imaginary part is close to 0, set it to 0
    #     if np.abs(np.imag(C)) < 1e-10:
    #         C = np.real(C)
    #
    #
    #     # Initialize the summation result
    #     P_ej_k = 0
    #
    #     # Compute the summation
    #     print(hyperedge.get_nodes())
    #     for l in range(n):
    #         C_l = C ** l
    #         product = 1
    #         for m in range(1, n):
    #                 print(states[hyperedge.get_nodes()[m]])
    #                 print(C_l)
    #                 print(product)
    #                 print()
    #                 product *= 1 + ((C_l - 1) * states[hyperedge.get_nodes()[m]])
    #
    #         P_ej_k += (C_l ** -k) * product
    #
    #     exit()
    #
    #
    #     # Compute the result
    #     P_ej_k /= n+1
    #
    #     P_ej_k = np.real(P_ej_k)
    #
    #     # print(P_ej_k)
    #     if P_ej_k == None or P_ej_k < 0 or P_ej_k > 1:
    #         print("m: ", m)
    #         print("hyperedge nodes: ", hyperedge.get_nodes())
    #         print("states: ", states)
    #         print("hyperedges: ", self.hypergraph.get_hyperedges())
    #         print("i: ", i)
    #         print("n: ", n)
    #         print("k: ", k)
    #         print("l: ", l)
    #         print("C: ", C)
    #         print("product: ", product)
    #         print("result: ", P_ej_k)
    #         exit()
    #
    #     # Ensure the result is within [0, 1]
    #     P_ej_K_clamped = np.clip(P_ej_K_real, 0, 1)
    #
    #     return P_ej_K_clamped
    #
    # # Define the differential equation
    # def dy_dt(self, t, y, delta, lambda_, hyperedges):
    #     print (y)
    #     dy = [0 for i in range(len(y))]
    #     for i in range(len(y)):
    #         dy[i] = -delta * y[i]
    #         if y[i] < 1:
    #             # calculate the sum of the probabilities of the hyperedges that contain the node i
    #             sum = 0
    #             for hyperedge in hyperedges:
    #                 size_ej = hyperedge.get_size()
    #                 if i in hyperedge.get_nodes():
    #                     pej_sum = 0
    #                     theta = int(self.simulation_parameters[2] * size_ej)
    #                     for k in range(theta , size_ej + 1):
    #                         pej_sum += np.log2(size_ej) * self.P_ej_K_k(k, size_ej, y, i, hyperedge)
    #                     sum += pej_sum
    #             dy[i] += lambda_ * (1-y[i]) * sum
    #     return dy
    #
    # def calculate_ODE_function(self):
    #     # set parameters
    #     delta = self.simulation_parameters[5]
    #     lambda_ = self.simulation_parameters[4]
    #
    #
    #     # set initial conditions
    #     y0 = [0 for i in range(self.hypergraph.get_number_of_nodes())]
    #
    #     seed_size = 0.1
    #     seed_nodes = random.sample(self.hypergraph.get_nodes(), int(seed_size * self.hypergraph.get_number_of_nodes()))
    #
    #     for node in seed_nodes:
    #         y0[node] = 1
    #
    #     y = y0
    #
    #     dt = 1
    #     t = 0
    #     tf = 50
    #
    #     func_y = [np.mean(y)]
    #
    #     if 1:
    #         print("parameters: ", delta, lambda_)
    #         print("initial conditions: ", y0)
    #         print("number of edges: ", self.hypergraph.get_number_of_hyperedges())
    #
    #     while t < tf:
    #         # k1 = dt * self.dy_dt(t, y, delta, lambda_, self.hypergraph.get_hyperedges())
    #         # y2_tmp = [y[i] + k1[i] / 2 for i in range(len(y))]
    #         # k2 = dt * self.dy_dt((t + dt )/ 2,y2_tmp , delta, lambda_, self.hypergraph.get_hyperedges())
    #         # y3_tmp = [y[i] + k2[i] / 2 for i in range(len(y))]
    #         # k3 = dt * self.dy_dt(t + dt / 2, y3_tmp, delta, lambda_, self.hypergraph.get_hyperedges())
    #         # y4_tmp = [y[i] + k3[i] for i in range(len(y))]
    #         # k4 = dt * self.dy_dt(t + dt, y4_tmp, delta, lambda_, self.hypergraph.get_hyperedges())
    #         #
    #         # y = [y[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(len(y))]
    #
    #         k1 = self.dy_dt(t, y, delta, lambda_, self.hypergraph.get_hyperedges())
    #
    #         y = [y[i] + k1[i] for i in range(len(y))]
    #
    #         t += dt
    #         func_y.append(np.mean(y))
    #         if t > 3:
    #             exit()
    #         # if 1:
    #         #     # print("t: ", t, " y: ", y)
    #         #     # print("k1: ", k1)
    #
    #     return func_y
