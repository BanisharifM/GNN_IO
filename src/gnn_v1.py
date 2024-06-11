import torch
from torch_geometric.data import Data
import json
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# Load your JSON data
with open("graph_json/graphs.json") as f:
    graph_data = json.load(f)

# Example function to convert your JSON graph to PyTorch Geometric Data
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

    # Print debug information
    print(f"Graph {graph['index']}:")
    print(f"  Number of nodes: {x.size(0)}")
    print(f"  Number of edges: {edge_index.size(1)}")
    print(f"  Node features: {x}")
    print(f"  Edge indices: {edge_index}")

    data = Data(x=x, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, y=y)
    return data

# Convert all graphs
data_list = [json_to_data(graph) for graph in graph_data]

# Example of how to use the data_list
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data_list[0].num_node_features, 16)
        self.conv2 = GCNConv(16, 2)  # Assuming 2 classes for simplicity, adjust as needed

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data_list[0])  # Assuming single graph for training, adjust as needed
    loss = F.nll_loss(out[data_list[0].train_mask], data_list[0].y[data_list[0].train_mask])
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    model.eval()
    _, pred = out.max(dim=1)
    correct = float (pred[data_list[0].train_mask].eq(data_list[0].y[data_list[0].train_mask]).sum().item())
    acc = correct / data_list[0].train_mask.sum().item()
    print(f'Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

def test():
    model.eval()
    logits, accs = model(data_list[0]), []
    for _, mask in data_list[0]("train_mask", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data_list[0].y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

for epoch in range(200):
    train()
    test_acc = test()
    print(f"Epoch {epoch:03d}, Test: {test_acc}")

torch.save(model.state_dict(), "model.pt")

