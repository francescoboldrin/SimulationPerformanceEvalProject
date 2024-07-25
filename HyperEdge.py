# contains the clas hyperedge

class Hyperedge:

    def __init__(self, size):
        self.size = size
        self.nodes = []

    # def __init_subclass__(cls, nodes):
    #     super().__init_subclass__()
    #     cls.nodes = nodes

    def __str__(self):
        return "Hyperedge: " + str(self.nodes)

    def __repr__(self):
        return "Hyperedge: " + str(self.nodes)

    def get_nodes(self):
        return self.nodes

    def set_nodes(self, nodes):
        self.nodes = nodes

    def get_size(self):
        return len(self.nodes)

    def reset_size(self):
        self.size = len(self.nodes)

    def add_node(self, node):
        self.nodes.append(node)

    def contains_node(self, node):
        # O(n) complexity
        # if sorted then we can do dichotomic search O(log(n))
        for n in self.nodes:
            if n == node:
                return True
        return False
