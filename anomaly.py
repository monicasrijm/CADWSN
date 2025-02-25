import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import networkx as nx

# Generate random sensor network graph
num_nodes = 10  # Number of sensor nodes
p_connect = 0.3  # Probability of edge creation
G = nx.erdos_renyi_graph(num_nodes, p_connect)

# Generate random sensor data for each node
data = {f'sensor{i}': np.random.normal(0, 1, 100) for i in range(num_nodes)}
df = pd.DataFrame(data)

# Normalize the data
df = (df - df.mean()) / df.std()

# Train Isolation Forest model for each node
models = {}
for node in G.nodes():
    models[node] = IsolationForest(contamination=0.1, random_state=42)
    models[node].fit(df[f'sensor{node}'].values.reshape(-1, 1))

# Predict anomalies for each node
for node in G.nodes():
    df[f'anomaly{node}'] = models[node].predict(df[f'sensor{node}'].values.reshape(-1, 1))

# Identify anomalous nodes
anomalous_nodes = [node for node in G.nodes() if np.any(df[f'anomaly{node}'] == -1)]

# Print anomalous nodes
print(f"Anomalous nodes: {anomalous_nodes}")

# Plot the sensor network graph with anomalies highlighted
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
nx.draw_networkx_nodes(G, pos, nodelist=anomalous_nodes, node_color='red', node_size=500)
plt.title('Sensor Network with Anomalous Nodes Highlighted')
plt.show()
