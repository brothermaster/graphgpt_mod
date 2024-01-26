import torch
import json
graph_data_path = "./graph_data/all_graph_data.pt"
graph_data_all = torch.load(graph_data_path)
data_path = "./data/stage_1/graph_matching.json"
list_data_dict = json.load(open(data_path, "r"))
print()