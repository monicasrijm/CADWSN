import numpy as np
import networkx as nx
import random

class SensorNode:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.neighbors = []
        self.data = None

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def send_data(self, base_station):
        if base_station in self.neighbors:
            self.data = random.randint(0, 100)
            return True
        return False

class BaseStation:
    def __init__(self, position):
        self.position = position
        self.received_data = []

    def receive_data(self, data):
        self.received_data.append(data)

class Network:
    def __init__(self, num_nodes, base_station_position):
        self.nodes = [SensorNode(i, (random.uniform(0, 10), random.uniform(0, 10))) for i in range(num_nodes)]
        self.base_station = BaseStation(base_station_position)
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        for node in self.nodes:
            for neighbor in self.nodes:
                if node != neighbor and np.linalg.norm(np.array(node.position) - np.array(neighbor.position)) < 2:
                    self.graph.add_edge(node.id, neighbor.id)
                    node.add_neighbor(neighbor)
                    neighbor.add_neighbor(node)

    def route_data(self):
        for node in self.nodes:
            if node.send_data(self.base_station):
                self.base_station.receive_data(node.data)

    def check_routing_integrity(self):
        for node in self.nodes:
            if len(node.neighbors) == 0:
                print(f"Node {node.id} has no neighbors, potential integrity issue!")
            elif len(node.neighbors) > 3:
                print(f"Node {node.id} has too many neighbors, potential integrity issue!")

    def simulate(self):
        # Generate random data for each node
        for node in self.nodes:
            node.data = random.randint(0, 100)

        self.route_data()
        self.check_routing_integrity()

# Example usage
num_nodes = 10
base_station_position = (5, 5)
network = Network(num_nodes, base_station_position)
network.simulate()

# Print the received data at the base station
print("Received data at the base station:", network.base_station.received_data)
