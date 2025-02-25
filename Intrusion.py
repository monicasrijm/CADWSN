import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestClassifier

class SensorNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.is_intruder = False

    def send_data(self):
        if self.is_intruder:
            return np.random.rand(10) * 10  # Simulate intruder data
        else:
            return np.random.rand(10)  # Simulate normal data

class NetworkSimulator:
    def __init__(self, num_nodes):
        self.nodes = [SensorNode(i) for i in range(num_nodes)]
        self.intrusion_probability = 0.1
        self.classifier = RandomForestClassifier()
        self.graph = nx.Graph()

    def simulate_intrusion(self):
        for node in self.nodes:
            if random.random() < self.intrusion_probability:
                node.is_intruder = True

    def generate_training_data(self):
        data = []
        labels = []
        for node in self.nodes:
            node_data = node.send_data()
            label = 1 if node.is_intruder else 0
            data.append(node_data)
            labels.append(label)
        return np.array(data), np.array(labels)

    def train_model(self):
        data, labels = self.generate_training_data()
        self.classifier.fit(data, labels)

    def detect_intrusions(self):
        data, _ = self.generate_training_data()
        predictions = self.classifier.predict(data)
        intruder_nodes = [self.nodes[i].node_id for i in range(len(predictions)) if predictions[i] == 1]
        return intruder_nodes

    def simulate_network(self):
        self.simulate_intrusion()
        self.train_model()
        detected_intrusions = self.detect_intrusions()
        print("Intruder nodes detected:", detected_intrusions)
        self.visualize_network(detected_intrusions)

    def visualize_network(self, intruder_nodes):
        self.graph.clear()
        for node in self.nodes:
            self.graph.add_node(node.node_id)
        for i in range(len(self.nodes) - 1):
            self.graph.add_edge(self.nodes[i].node_id, self.nodes[i+1].node_id)

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='green')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=intruder_nodes, node_color='red')
        plt.show()

# Example usage
simulator = NetworkSimulator(num_nodes=10)
simulator.simulate_network()
