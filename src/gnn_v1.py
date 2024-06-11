import torch
from torch_geometric.data import Data
import json

# Load your JSON data
with open("graph_json/graphs.json") as f:
    graph_data = json.load(f)

# Example function to convert your JSON graph to PyTorch Geometric Data
def json_to_data(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Filter out nodes with empty values and convert to tensor
    x = torch.tensor([node["value"] for node in nodes if node["value"] != ""], dtype=torch.float)
    edge_index = (
        torch.tensor(
            [[edge["source"], edge["target"]] for edge in edges], dtype=torch.long
        )
        .t()
        .contiguous()
    )

    data = Data(x=x, edge_index=edge_index)
    return data

# Check the structure of the JSON data
if "graphs" in graph_data and isinstance(graph_data["graphs"], list):
    data_list = [json_to_data(graph) for graph in graph_data["graphs"]]
else:
    raise ValueError("The JSON structure is not as expected. 'graphs' key is missing or not a list.")

# Example of how to use the data_list
# Here, we assume you will use the first graph for training

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
    print(f"Epoch {epoch:03d}, Test: {test_acc:.4f}")

torch.save(model.state_dict(), "model.pt")

import shap

# Assuming your model and data are ready
model.eval()
background = data_list[:100]  # Background dataset for SHAP
explainer = shap.KernelExplainer(model, background)

# Test instance
test_instance = data_list[100]
shap_values = explainer.shap_values(test_instance)

# Visualize the SHAP values
shap.summary_plot(shap_values, test_instance)

