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

    # Assuming you have node features and edge features
    x = torch.tensor([node["value"] for node in nodes], dtype=torch.float)
    edge_index = (
        torch.tensor(
            [[edge["source"], edge["target"]] for edge in edges], dtype=torch.long
        )
        .t()
        .contiguous()
    )

    data = Data(x=x, edge_index=edge_index)
    return data


# Convert all graphs
data_list = [json_to_data(graph) for graph in graph_data["graphs"]]


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

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
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data("train_mask ", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
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
