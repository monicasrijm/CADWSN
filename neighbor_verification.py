import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class SensorNode:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

class Network:
    def __init__(self, num_nodes):
        self.nodes = [SensorNode(i, (random.uniform(0, 10), random.uniform(0, 10))) for i in range(num_nodes)]
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        for node in self.nodes:
            for neighbor in self.nodes:
                if node != neighbor and np.linalg.norm(np.array(node.position) - np.array(neighbor.position)) < 3:
                    self.graph.add_edge(node.id, neighbor.id)
                    node.add_neighbor(neighbor)
                    neighbor.add_neighbor(node)

    def verify_neighbors(self):
        inconsistent_nodes = []
        for node in self.nodes:
            actual_neighbors = set(neigh.id for neigh in node.neighbors)
            expected_neighbors = set(self.graph.neighbors(node.id))
            
            if actual_neighbors != expected_neighbors:
                print(f"Node {node.id} has inconsistent neighbors:")
                print(f"  Actual neighbors: {actual_neighbors}")
                print(f"  Expected neighbors: {expected_neighbors}")
                inconsistent_nodes.append(node.id)
        
        return inconsistent_nodes

    def simulate(self):
        return self.verify_neighbors()

    def visualize(self, inconsistent_nodes):
        pos = {node.id: node.position for node in self.nodes}
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=inconsistent_nodes, node_color='red', node_size=500)
        plt.title('Sensor Network with Inconsistent Nodes Highlighted')
        plt.show()

# Example usage
num_nodes = 10
network = Network(num_nodes)
inconsistent_nodes = network.simulate()
network.visualize(inconsistent_nodes)
