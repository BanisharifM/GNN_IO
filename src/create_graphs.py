import os
import pandas as pd
import networkx as nx
import json

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
output_dir = "graph_json/v4"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created at: {output_dir}")

# Function to create and save graph for each row
def create_and_save_graph(row, index, chunk_index, json_file_path):
    G = nx.DiGraph()
    root_node = "Tag"
    G.add_node(root_node, value=row["tag"])

    for category, counters in categories.items():
        G.add_node(category, value="")
        G.add_edge(root_node, category, weight=1)  # Edge from root to category with weight 1
        for counter in counters:
            if counter in row:
                G.add_node(counter, value=row[counter])
                # Calculate weight for the edge as the node's value
                weight = float(row[counter]) if row[counter] else 0
                G.add_edge(category, counter, weight=weight)

    # Convert the graph to a dictionary representation
    graph_dict = {
        "index": index,
        "nodes": [
            {"id": node, "value": data["value"]} for node, data in G.nodes(data=True)
        ],
        "edges": [
            {"source": source, "target": target, "weight": data["weight"]}
            for source, target, data in G.edges(data=True)
        ],
    }

    # Append the graph dictionary to the JSON file
    with open(json_file_path, "a") as f:
        json.dump(graph_dict, f)
        f.write('\n')

    print(f"Graph {index} saved to {json_file_path}")

# Load the CSV file in chunks
file_path = "data_csv/sample_train.csv"
chunk_size = 100000

for chunk_index, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    json_file_path = f"{output_dir}/graphs_chunk_{chunk_index}.json"
    if os.path.exists(json_file_path):
        os.remove(json_file_path)
    # Create and save graphs for the current chunk
    for index, row in chunk.iterrows():
        create_and_save_graph(row, index, chunk_index, json_file_path)

