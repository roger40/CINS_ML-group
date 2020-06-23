# import numpy as np
# import networkx as nx
# import os
"""
   PGLP is population generation via label propagation
   by lanlan Yu in 2020/5/13
"""
class PGLP(object):

    Chromosomes = None  # populations for clustering

    def __init__(self, pop_size, graph):
        """
        get initialization of population via label propagation
        :param pop_size:
        :param graph:
        """
        # pointed out in White,  D.  R.  &  Harary,  F.  (2001) Sociol.  Methodol.  31, 305–359.
        iters = 5

        self.nodes = list(graph.nodes)
        node_num = len(self.nodes)
        self.Chromosomes = []
        assignment = [i for i in range(node_num)]

        for i in range(pop_size):  # for each population
            for j in range(iters):  # iterative
                assignment = self.propagation(assignment, graph)  # label propagation
            self.Chromosomes.append(assignment)

    def propagation(self, labels, graph):
        node_num = len(self.nodes)
        for i in range(node_num):
            node = self.nodes[i]
            neighbors = list(graph.neighbors(node))
            count = {}
            for neighbor in neighbors:
                index = self.nodes.index(neighbor)
                count[labels[index]] = count.get(labels[index], 0) + 1
            label_sorted = sorted(count.items(), key=lambda x: x[1], reverse=True)
            labels[i] = label_sorted[0][0]

        return labels

# # for test
#
# if __name__ == '__main__':
#     import networkx as nx
#     graph = nx.Graph()
#     graph.add_edge(1, 2)
#     graph.add_edge(3, 4)
#     p = PGLP(2, graph)
#     print(p.Chromosomes)