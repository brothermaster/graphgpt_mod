from ogb.graphproppred import PygGraphPropPredDataset
dataset = PygGraphPropPredDataset(name="ogbg-molpcba")
 # Pytorch Geometric dataset object
split_idx = dataset.get_idx_split()
 # Dictionary containing train/valid/test indices.
train_idx = split_idx["train"]
