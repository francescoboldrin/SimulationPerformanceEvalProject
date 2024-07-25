# definition of the class node

class Node:
        # in future work this class can be erased since it amplifies the size of the hypergraph in terms of efficiency it downgrades the performance
        def __init__(self, id):
            self.id = id

            # used only in the generator, then is kind of useless and should be removed
            self.degree = 0
            self.hyperedges = []
        def __str__(self):
            return "Node: " + str(self.id) + " with degree: " + str(self.degree)

        def __repr__(self):
            return "Node: " + str(self.id) + " with degree: " + str(self.degree)

        def get_id(self):
            return self.id

        def get_degree(self):
            return self.degree

        def get_hyperedges(self):
            return self.hyperedges

        def set_degree(self, degree):
            self.degree = degree

        def set_hyperedges(self, hyperedges):
            self.hyperedges = hyperedges

        def add_hyperedge(self, hyperedge):
            self.hyperedges.append(hyperedge)

        def reset_hyperedges(self):
            self.hyperedges = []

        def reset_degree(self):
            self.degree = len(self.hyperedges)