# ------train, created and maintained by longma2------

import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import SAGEConv

from load_data import load_data
from preprocess_data import CiteseerDataset
from preprocess_data import list2array
from preprocess_data_facebook import FacebookDataset
# from preprocess_data_facebook import list2array
# from load_data_facebook import load_data

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats1, h_feats2, h_feats3, h_feats4, h_feats5, h_feats6, h_feats7, h_feats8, h_feats9, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats1)
        self.conv2 = GraphConv(h_feats1, h_feats2)
        self.conv3 = GraphConv(h_feats2, h_feats3)
        self.conv4 = GraphConv(h_feats3, h_feats4)
        self.conv5 = GraphConv(h_feats4, h_feats5)
        self.conv6 = GraphConv(h_feats5, h_feats6)
        self.conv7 = GraphConv(h_feats6, h_feats7)
        self.conv8 = GraphConv(h_feats7, h_feats8)
        self.conv9 = GraphConv(h_feats8, h_feats9)
        self.conv10 = GraphConv(h_feats9, num_classes)



    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        h = F.relu(h)
        h = self.conv6(g, h)
        h = F.relu(h)
        h = self.conv7(g, h)
        h = F.relu(h)
        h = self.conv8(g, h)
        h = F.relu(h)
        h = self.conv9(g, h)
        h = F.relu(h)
        h = self.conv10(g, h)

        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(1001):
        # Forward
        logits = model(g, features)
        # print(logits)
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

        with open("original_output_citeseer.txt", "a") as f:
            if e % 5 == 0:
                f.write('In epoch {}, loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})\n'.format(
                    e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))




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


    # path_edge = './facebook_large/musae_facebook_edges.csv'
    # path_node_labels = './facebook_large/musae_facebook_target.csv'
    # path_node_features = './facebook_large/musae_facebook_features.json'
    # _edge_list, node_labels_list, node_features_list = load_data(path_edge=path_edge, path_node_labels=path_node_labels, path_node_features = path_node_features)
    # edge_list = []
    # for edge in _edge_list:
    #     edge_list.append(edge)
    #     edge_list.append([edge[1], edge[0]])
    #
    # N_nodes, N_edges, node_features_ndarray, node_labels_ndarray, edge_features_ndarray, edges_src_ndarray, edges_dst_ndarray = list2array(node_labels_list=node_labels_list, edge_list=edge_list, node_features_list=node_features_list)
    # dataset = FacebookDataset(N_nodes=N_nodes, N_edges=N_edges, node_features_ndarray=node_features_ndarray, node_labels_ndarray=node_labels_ndarray, edge_features_ndarray=edge_features_ndarray, edges_src_ndarray=edges_src_ndarray, edges_dst_ndarray=edges_dst_ndarray)
    # graph = dataset[0]
    # graph = dgl.add_self_loop(graph)


    if len(graph.ndata['feat'].shape) == 1: in_feat = 1
    else: in_feat = graph.ndata['feat'].shape[1]

    # Create the model with given dimensions
    model = GCN(in_feat, 64, 128, 256, 512, 256, 128, 64, 32, 16, dataset.num_classes)
    train(graph, model)

    # dataset = dgl.data.CoraGraphDataset()
    # g = dataset[0]
    # model = GCN(g.ndata['feat'].shape[1], 64, 128, 256, 64, 16, dataset.num_classes)
    # train(g, model)
