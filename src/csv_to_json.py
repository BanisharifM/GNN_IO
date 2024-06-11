import os
import pandas as pd
import networkx as nx
import json

# Load the CSV file
file_path = "data_csv/sample_train_100.csv"
df = pd.read_csv(file_path)

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

# Path to the JSON file
json_file_path = f"{output_dir}/graphs.json"

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


# Create and save graphs for the first 10 rows
for index, row in df.iterrows():
    if index < 100:
        create_and_save_graph(row, index)
    else:
        break

# Save all graphs to the JSON file
with open(json_file_path, "w") as f:
    json.dump(graphs, f, indent=4)

print(f"Graphs saved to JSON file at: {json_file_path}")
