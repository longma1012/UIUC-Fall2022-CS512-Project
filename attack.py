import dgl
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE
from scipy.sparse import csr_matrix
from train import GCN
from train import train


dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
n = g.num_nodes()




original_adj = g.adj().to_dense().numpy()
adj = csr_matrix(original_adj)

feats = g.ndata['feat'].numpy()

labels = g.ndata['label'].numpy()

model = DICE()

model.attack(adj, labels, n_perturbations=10)
modified_adj = model.modified_adj
coo_mod_adj = modified_adj.tocoo()
rows = []
cols = []
for i in coo_mod_adj.row:
    rows.append(i)
for i in coo_mod_adj.col:
    cols.append(i)

edges = []
for i in range(len(rows)):
    edges.append((rows[i], cols[i]))

graph = dgl.DGLGraph()
graph.add_nodes(n)
graph.add_edge(rows, cols)


if len(graph.ndata['feat'].shape) == 1: in_feat = 1
else: in_feat = graph.ndata['feat'].shape[1]
model = GCN(in_feat, 64, 128, 256, 512, 256, 128, 64, 32, 16, dataset.num_classes)
train(graph, model)
