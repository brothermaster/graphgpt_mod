import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import dgl
from torch.nn import MultiheadAttention
import urllib.request

import pandas as pd
from torch_geometric.nn.models import GAT as PGGAT
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import NeighborSampler, Data
# gat = PGGAT(in_channels=109,
#             hidden_channels=512,
#             out_channels=512,
#             heads=4,
#             concat=True,
#             dropout=0.2,
#             negative_slope=0.2,
#             num_layers=2,
#             act="prelu",
#             norm="batch_norm")

# urllib.request.urlretrieve(
#     "https://data.dgl.ai/tutorial/dataset/members.csv", "./members.csv"
# )
# urllib.request.urlretrieve(
#     "https://data.dgl.ai/tutorial/dataset/interactions.csv",
#     "./interactions.csv",
# )
# nn.TransformerDecoderLayer()
# multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
# # attn_output, attn_output_weights = multihead_attn(query, key, value)

class my_data_set:

    def __init__(self):
        g1 = dgl.graph((torch.tensor([0, 1, 2, 3]), torch.tensor([0, 0 ,0,0])))
        g1.ndata['feat'] = torch.ones(4,10)
        g1.ndata['label'] = torch.tensor([[0,1,0],[1,0,1],[1,1,1],[0,1,1]])
        g1 = dgl.add_self_loop(g1)
        g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
        g2.ndata['feat'] = torch.ones(3,10)
        g2.ndata['label'] = torch.tensor([[0,1,0],[1,0,1],[1,1,1]])
        g2 = dgl.add_self_loop(g2)
        self.bg = [g1, g2]
        self.label = torch.tensor([[0,1,0],[1,0,1]])

    def __getitem__(self, index: int):
        g_i = self.bg[index]
        l_i = self.label[index]
        return g_i,l_i
    
    def __len__(self):
        """ returns length of data """
        return len(self.bg)
    
    @property
    def num_labels(self):
        return 3
    
    @staticmethod
    def collate_fn(batch):
        # graphs = [x[0].add_self_loop() for x in batch]
        graphs = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        batch_g = dgl.batch(graphs)
        labels = torch.stack(labels,dim=0)
        # labels = torch.cat(labels, dim=0)
        return batch_g, labels

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()

        perious_h = heads[0]
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, perious_h, activation=F.elu)
        )

        for h in heads[1:]:
            self.gat_layers.append(
                dglnn.GATConv(
                    hid_size * perious_h,
                    hid_size,
                    h,
                    residual=True,
                    activation=F.elu,
                )
            )
            perious_h = h

        self.liner = nn.Linear(hid_size*h,out_size)

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h) 
            h = h.flatten(-2)

        h = self.liner(h)
        g.ndata['h'] = h
        # Calculate graph representation by average readout.
        hg = dgl.mean_nodes(g, 'h')
        return hg
    
def train(train_dataloader, device, model):
    # define loss function and optimizer
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)

    # training loop
    for epoch in range(400):
        model.train()
        logits = []
        total_loss = 0
        # mini-batch loop
        for batch_id, (batched_graph,batched_labels) in enumerate(train_dataloader):
            labels = batched_labels.float().to(device)
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata["feat"].float()
            logits = model(batched_graph, features)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            "Epoch {:05d} | Loss {:.4f} |".format(
                epoch, total_loss / (batch_id + 1)
            )
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load and preprocess datasets
    train_dataset = my_data_set()
    # features = train_dataset[0].ndata["feat"]
    # create GAT model
    in_size = 10
    out_size = train_dataset.num_labels
    model = GAT(in_size, 256, out_size, heads=[4, 4, 6]).to(device)

    # model training
    print("Training...")
    # train_dataloader = GraphDataLoader(train_dataset, batch_size=2)
    train_dataloader = DataLoader(train_dataset, batch_size=2,
                                  shuffle= False,collate_fn=train_dataset.collate_fn)
    train(train_dataloader, device, model)
