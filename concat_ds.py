import json
import random
import re
import pandas as pd
from tqdm import tqdm
import torch as th
from torch_geometric.utils import subgraph
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.utils import from_scipy_sparse_matrix
import logging
import copy

# 数据集名称
dsname ='addtasks'
stage = "pretrain" # zeroshots
tasks = ['income','car']

for split_type in ['train','val','test']:
    try:
        cat = []
        for task in tasks:
            with open(f'./instruct_ds/{dsname}/{stage}_{task}_{split_type}.json') as f:
                tmp = json.load(f)
                cat = cat + tmp

        with open(f'./instruct_ds/{dsname}/{stage}_{"-".join(tasks)}_{split_type}.json', 'w') as f:
            json.dump(cat, f)
    except Exception as e:
        print(e)