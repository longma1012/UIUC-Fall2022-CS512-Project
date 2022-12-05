# ------train, created and maintained by longma2------

import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

from preprocess_data import CiteseerDataset
from preprocess_data import list2array
from load_data import load_data

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(500):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # print(logits[train_mask].shape)
        # print(labels[train_mask].shape)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))




if __name__ == '__main__':

    path_edge = './citeseer_edges.txt'
    path_node_labels = './citeseer_node_labels.txt'

    _edge_list, node_labels_list = load_data(path_edge, path_node_labels)
    edge_list = []
    for edge in _edge_list:
        edge_list.append(edge)
        edge_list.append([edge[1], edge[0]])
    N_nodes, N_edges, node_indexes, node_labels_ndarray, edge_features_ndarray, edges_src_ndarray, edges_dst_ndarray = list2array(node_labels_list=node_labels_list, edge_list=edge_list)
    dataset = CiteseerDataset(N_nodes=N_nodes, N_edges=N_edges, node_indexes=node_indexes, node_labels_ndarray=node_labels_ndarray, edge_features_ndarray=edge_features_ndarray, edges_src_ndarray=edges_src_ndarray, edges_dst_ndarray=edges_dst_ndarray)
    graph = dataset[0]

    if len(graph.ndata['feat'].shape) == 1: in_feat = 1
    else: in_feat = graph.ndata['feat'].shape[1]

    # Create the model with given dimensions
    model = GCN(in_feat, 16, dataset.num_classes)
    train(graph, model)

    # dataset = dgl.data.CoraGraphDataset()
    # g = dataset[0]
    # model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
    # train(g, model)
