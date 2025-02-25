# import numpy as np
# import pandas as pd
# import networkx as nx
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # Generate a sample dataset for network traffic
# def generate_data(num_samples):
#     data = {
#         'Energy': np.random.rand(num_samples) * 100,
#         'Traffic': np.random.randint(50, 200, size=num_samples),
#         'Packet_Size': np.random.randint(20, 100, size=num_samples),
#         'Signal_Strength': np.random.rand(num_samples) * 50,
#         'Label': np.random.choice([0, 1], size=num_samples)
#     }
#     return pd.DataFrame(data)

# # Create a random network topology
# def create_network(num_nodes):
#     G = nx.erdos_renyi_graph(num_nodes, 0.3)
#     return G

# # Anomaly detection using Random Forest
# def train_model(df):
#     X = df[['Energy', 'Traffic', 'Packet_Size', 'Signal_Strength']]
#     y = df['Label']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#     print("Classification Report:")
#     print(report)
#     return clf

# # Simulate network traffic and perform real-time anomaly detection
# def simulate_network_traffic(G, clf):
#     for node in G.nodes:
#         data_point = [
#             np.random.rand() * 100,  # Energy
#             np.random.randint(50, 200),  # Traffic
#             np.random.randint(20, 100),  # Packet_Size
#             np.random.rand() * 50  # Signal_Strength
#         ]
#         prediction = clf.predict([data_point])
#         if prediction == 1:
#             print(f"Anomaly detected at node {node}: Potential Cyber Attack")
#         else:
#             print(f"Node {node} - Normal behavior detected")

# # Main function to run the simulation
# def main():
#     num_samples = 1000
#     num_nodes = 10

#     # Generate dataset and train the model
#     df = generate_data(num_samples)
#     clf = train_model(df)

#     # Create network and simulate traffic
#     G = create_network(num_nodes)
#     simulate_network_traffic(G, clf)

# if __name__ == "__main__":
#     main()


#  #TRIAL 2

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Generate a sample dataset and train the model (for illustration purposes)
def generate_data(num_samples):
    data = {
        'Energy': np.random.rand(num_samples) * 100,
        'Traffic': np.random.randint(50, 200, size=num_samples),
        'Packet_Size': np.random.randint(20, 100, size=num_samples),
        'Signal_Strength': np.random.rand(num_samples) * 50,
        'Label': np.random.choice([0, 1], size=num_samples)
    }
    return pd.DataFrame(data)

def train_model(df):
    X = df[['Energy', 'Traffic', 'Packet_Size', 'Signal_Strength']]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    return clf

# Callback function for MQTT data reception
def on_message(client, userdata, message):
    data_point = np.frombuffer(message.payload, dtype=np.float64)
    prediction = clf.predict([data_point])
    if prediction == 1:
        print("Anomaly detected: Potential Cyber Attack")
    else:
        print("Normal behavior detected")

# Main function to run the real-time data pipeline
def main():
    num_samples = 1000

    # Generate dataset and train the model
    df = generate_data(num_samples)
    global clf
    clf = train_model(df)

    # Set up MQTT client and connect to the broker
    client = mqtt.Client()
    client.on_message = on_message
    client.connect("mqtt_broker_address", 1883, 60)  # Replace with your MQTT broker address

    # Subscribe to the topic where WSN nodes publish data
    client.subscribe("wsn/data")

    # Start the MQTT client loop
    client.loop_forever()

if __name__ == "__main__":
    main()
