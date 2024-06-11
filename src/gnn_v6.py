import os
import json
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Directory where the graph JSON files are stored
input_dir = "graph_json/v6"

# Function to convert JSON graph to PyTorch Geometric Data
def json_to_data(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Create a mapping from node IDs to integers
    node_id_map = {node["id"]: i for i, node in enumerate(nodes)}

    # Filter out nodes with empty values
    valid_nodes = [node for node in nodes if node["value"] != ""]
    node_values = [node["value"] for node in valid_nodes]
    node_id_map = {node["id"]: i for i, node in enumerate(valid_nodes)}  # Update mapping with valid nodes only

    if len(node_values) == 0:
        raise ValueError("All node values are empty. Cannot create tensor.")

    # Convert node values to tensor
    x = torch.tensor(node_values, dtype=torch.float).view(-1, 1)  # Ensure it has shape [num_nodes, num_features]

    # Convert edges to use the integer mapping
    edge_index = torch.tensor(
        [[node_id_map[edge["source"]], node_id_map[edge["target"]]] for edge in edges], dtype=torch.long
    ).t().contiguous()

    # Create random masks for training, validation, and testing
    num_nodes = len(node_values)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_indices = random.sample(range(num_nodes), int(0.6 * num_nodes))
    val_indices = random.sample([i for i in range(num_nodes) if i not in train_indices], int(0.2 * num_nodes))
    test_indices = [i for i in range(num_nodes) if i not in train_indices and i not in val_indices]

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Create labels for the nodes
    y = torch.zeros(num_nodes, dtype=torch.float)
    y[0] = graph["nodes"][0]["value"]  # Assuming the first node is the "Tag" node

    data = Data(x=x, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, y=y)
    return data

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 1)  # Predicting a single value (regression)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Training function
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask].view(-1, 1))  # Mean Squared Error Loss
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    model.eval()
    out = model(data)
    mse = F.mse_loss(out[data.train_mask], data.y[data.train_mask].view(-1, 1)).item()
    print(f'Loss: {loss.item():.4f}, MSE: {mse:.4f}')

# Testing function
def test(data):
    model.eval()
    logits = model(data)
    mse = F.mse_loss(logits[data.test_mask], data.y[data.test_mask].view(-1, 1)).item()
    return mse

# Initialize the model and optimizer
model = None
optimizer = None

# Load each chunk JSON file, convert it to data, and train the model
json_files = [f for f in os.listdir(input_dir) if f.startswith('graphs_chunk_') and f.endswith('.json')]

if not json_files:
    raise ValueError(f"No JSON files found in the directory {input_dir}")

for json_file in json_files:
    data_list = []
    json_file_path = os.path.join(input_dir, json_file)
    print(f"Loading file: {json_file_path}")
    with open(json_file_path, "r") as f:
        for line in f:
            graph = json.loads(line)
            try:
                data = json_to_data(graph)
                data_list.append(data)
            except KeyError as e:
                print(f"Skipping graph due to missing node: {e}")

    if not data_list:
        print(f"File {json_file} has no valid graphs with edges. Skipping training for this file.")
        continue

    if model is None:
        model = GNN(data_list[0].num_node_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training loop for the current chunk
    for epoch in range(200):  # Set the number of epochs to 200
        for data in data_list:
            train(data)
        mse = test(data)
        print(f"File {json_file}, Epoch {epoch:03d}, MSE: {mse:.4f}")

if model is not None:
    # Save the trained model
    torch.save(model.state_dict(), "model_v6.pt")
    print("Model saved successfully.")
else:
    print("No valid data was loaded; model was not trained.")

