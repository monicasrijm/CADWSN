import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestClassifier

class SensorNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.attack_type = None

    def send_data(self):
        if self.attack_type == "DoS":
            return np.random.rand(10) * 20  # Simulate DoS attack data
        elif self.attack_type == "Data Tampering":
            return np.random.rand(10) * 10  # Simulate Data Tampering attack data
        elif self.attack_type == "Eavesdropping":
            return np.random.rand(10) * 5   # Simulate Eavesdropping attack data
        else:
            return np.random.rand(10)  # Simulate normal data

class NetworkSimulator:
    def __init__(self, num_nodes):
        self.nodes = [SensorNode(i) for i in range(num_nodes)]
        self.attack_probability = 0.3
        self.classifier = RandomForestClassifier()
        self.graph = nx.Graph()

    def simulate_attack(self):
        attack_types = [None, "DoS", "Data Tampering", "Eavesdropping"]
        for node in self.nodes:
            node.attack_type = random.choices(attack_types, [1 - self.attack_probability, self.attack_probability/3, self.attack_probability/3, self.attack_probability/3])[0]

    def generate_training_data(self):
        data = []
        labels = []
        for node in self.nodes:
            node_data = node.send_data()
            label = node.attack_type if node.attack_type else "Normal"
            data.append(node_data)
            labels.append(label)
        return np.array(data), np.array(labels)

    def train_model(self):
        data, labels = self.generate_training_data()
        self.classifier.fit(data, labels)

    def detect_attacks(self):
        data, _ = self.generate_training_data()
        predictions = self.classifier.predict(data)
        compromised_nodes = {node.node_id: predictions[i] for i, node in enumerate(self.nodes) if predictions[i] != "Normal"}
        return compromised_nodes

    def simulate_network(self):
        self.simulate_attack()
        self.train_model()
        detected_attacks = self.detect_attacks()
        print("Compromised nodes detected:", detected_attacks)
        self.visualize_network(detected_attacks)

    def visualize_network(self, compromised_nodes):
        self.graph.clear()
        for node in self.nodes:
            self.graph.add_node(node.node_id)
        for i in range(len(self.nodes) - 1):
            self.graph.add_edge(self.nodes[i].node_id, self.nodes[i+1].node_id)

        pos = nx.spring_layout(self.graph)
        node_colors = ['red' if node_id in compromised_nodes else 'green' for node_id in range(len(self.nodes))]
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors)

        attack_labels = {node_id: attack_type for node_id, attack_type in compromised_nodes.items()}
        nx.draw_networkx_labels(self.graph, pos, labels=attack_labels, font_color='blue')
        plt.show()

# Example usage
simulator = NetworkSimulator(num_nodes=10)
simulator.simulate_network()
