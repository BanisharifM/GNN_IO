import os
import pandas as pd
import networkx as nx
import json
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define categories and their corresponding counters
categories = {
    "POSIX_ACCESS": [
        "POSIX_ACCESS1_ACCESS",
        "POSIX_ACCESS2_ACCESS",
        "POSIX_ACCESS3_ACCESS",
        "POSIX_ACCESS4_ACCESS",
        "POSIX_ACCESS1_COUNT",
        "POSIX_ACCESS2_COUNT",
        "POSIX_ACCESS3_COUNT",
        "POSIX_ACCESS4_COUNT",
    ],
    "POSIX_STRIDE": [
        "POSIX_STRIDE1_STRIDE",
        "POSIX_STRIDE2_STRIDE",
        "POSIX_STRIDE3_STRIDE",
        "POSIX_STRIDE4_STRIDE",
        "POSIX_STRIDE1_COUNT",
        "POSIX_STRIDE2_COUNT",
        "POSIX_STRIDE3_COUNT",
        "POSIX_STRIDE4_COUNT",
    ],
    "POSIX_SIZE": [
        "POSIX_SIZE_READ_0_100",
        "POSIX_SIZE_READ_100_1K",
        "POSIX_SIZE_READ_1K_10K",
        "POSIX_SIZE_READ_100K_1M",
        "POSIX_SIZE_WRITE_0_100",
        "POSIX_SIZE_WRITE_100_1K",
        "POSIX_SIZE_WRITE_1K_10K",
        "POSIX_SIZE_WRITE_10K_100K",
        "POSIX_SIZE_WRITE_100K_1M",
    ],
    "POSIX_OPERATIONS": [
        "POSIX_OPENS",
        "POSIX_FILENOS",
        "POSIX_READS",
        "POSIX_WRITES",
        "POSIX_SEEKS",
        "POSIX_STATS",
    ],
    "POSIX_DATA_TRANSFERS": ["POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN"],
    "POSIX_PATTERNS": [
        "POSIX_CONSEC_READS",
        "POSIX_CONSEC_WRITES",
        "POSIX_SEQ_READS",
        "POSIX_SEQ_WRITES",
    ],
    "POSIX_ALIGNMENTS": [
        "POSIX_MEM_ALIGNMENT",
        "POSIX_FILE_ALIGNMENT",
        "POSIX_MEM_NOT_ALIGNED",
        "POSIX_FILE_NOT_ALIGNED",
    ],
    "POSIX_RESOURCE_UTILIZATION": ["POSIX_RW_SWITCHES"],
    "LUSTRE_CONFIGURATION": ["LUSTRE_STRIPE_SIZE", "LUSTRE_STRIPE_WIDTH"],
    "GENERAL": ["nprocs"],
}

# Directory to save the graph representations
output_dir = "graph_json"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created at: {output_dir}")

# Initialize a list to store all graph details
graphs = []

# Function to create and save graph for each row
def create_and_save_graph(row, index):
    G = nx.DiGraph()
    root_node = "Tag"
    G.add_node(root_node, value=row["tag"])

    for category, counters in categories.items():
        G.add_node(category, value="")
        G.add_edge(root_node, category, data={})
        for counter in counters:
            if counter in row:
                G.add_node(counter, value=row[counter])
                G.add_edge(category, counter, data={})

    # Convert the graph to a dictionary representation
    graph_dict = {
        "index": index,
        "nodes": [
            {"id": node, "value": data["value"]} for node, data in G.nodes(data=True)
        ],
        "edges": [
            {"source": source, "target": target, "data": data["data"]}
            for source, target, data in G.edges(data=True)
        ],
    }

    # Append the graph dictionary to the list of graphs
    graphs.append(graph_dict)
    print(f"Graph {index} created")

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

    # Filter edges to only include those connecting valid nodes
    valid_edges = [edge for edge in edges if edge["source"] in node_id_map and edge["target"] in node_id_map]

    if len(valid_edges) == 0:
        print(f"Warning: Graph {graph['index']} has no valid edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # Convert edges to use the integer mapping
        edge_index = torch.tensor(
            [[node_id_map[edge["source"]], node_id_map[edge["target"]]] for edge in valid_edges], dtype=torch.long
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

    # Create random labels for the nodes
    y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, y=y)
    return data

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 2)  # Assuming 2 classes for simplicity, adjust as needed

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training function
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    model.eval()
    _, pred = out.max(dim=1)
    correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / data.train_mask.sum().item()
    print(f'Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

# Testing function
def test(data):
    model.eval()
    logits = model(data)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

# Load the CSV file in chunks
file_path = "data_csv/sample_train.csv"
chunk_size = 100000

model = None  # Initialize the model variable
optimizer = None  # Initialize the optimizer variable

for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    graphs = []  # Reset the graphs list for each chunk

    # Create and save graphs for the current chunk
    for index, row in chunk.iterrows():
        create_and_save_graph(row, index + i * chunk_size)

    # Save the current chunk's graphs to a JSON file
    json_file_path = f"{output_dir}/graphs_{i}.json"
    with open(json_file_path, "w") as f:
        json.dump(graphs, f, indent=4)

    # Load the current chunk's graphs and train the model
    data_list = [json_to_data(graph) for graph in graphs]

    # Initialize the model and optimizer after the first chunk is processed
    if model is None:
        model = GNN(data_list[0].num_node_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training loop for the current chunk
    for epoch in range(200):  # Adjust the number of epochs as needed
        for data in data_list:
            train(data)
        test_acc = test(data)
        print(f"Chunk {i}, Epoch {epoch:03d}, Test: {test_acc}")

# Save the trained model
torch.save(model.state_dict(), "model.pt")

