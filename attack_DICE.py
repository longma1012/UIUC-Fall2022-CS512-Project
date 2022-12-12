import dgl
import torch
import numpy as np
from deeprobust.graph.global_attack import DICE
from scipy.sparse import csr_matrix
from train import GCN
from train import train


dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
n_nodes = g.num_nodes()

node_features = g.ndata['feat']
node_labels = g.ndata['label']
# edge_features = g.edata['weight']



original_adj = g.adj().to_dense().numpy()
adj = csr_matrix(original_adj)

feats = g.ndata['feat'].numpy()

labels = g.ndata['label'].numpy()

model = DICE()
n_perturbations=0
model.attack(adj, labels, n_perturbations)
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
graph.add_nodes(n_nodes)
graph.add_edge(rows, cols)

graph.ndata['feat'] = node_features
graph.ndata['label'] = node_labels
# graph.edata['weight'] = edge_features


n_train = int(n_nodes * 0.6)
n_val = int(n_nodes * 0.2)
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
val_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[:n_train] = True
val_mask[n_train:n_train + n_val] = True
test_mask[n_train + n_val:] = True
graph.ndata['train_mask'] = train_mask
graph.ndata['val_mask'] = val_mask
graph.ndata['test_mask'] = test_mask
graph = dgl.add_self_loop(graph)

if len(graph.ndata['feat'].shape) == 1: in_feat = 1
else: in_feat = graph.ndata['feat'].shape[1]

model = GCN(in_feat, 64, 128, 256, 512, 256, 128, 64, 32, 16, dataset.num_classes)
train(graph, model,result_path='results/DICEattack'+str(n_perturbations)+'.txt')
