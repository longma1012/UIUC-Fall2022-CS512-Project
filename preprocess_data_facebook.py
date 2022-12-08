import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
import os
import csv

from load_data_facebook import load_data

class FacebookDataset(DGLDataset):
    def __init__(self, N_nodes, N_edges, node_features_ndarray, node_labels_ndarray, edge_features_ndarray, edges_src_ndarray, edges_dst_ndarray):
        self.N_nodes = N_nodes
        self.N_edges = N_edges
        self.node_features_ndarray = node_features_ndarray
        self.node_labels_ndarray = node_labels_ndarray
        self.edge_features_ndarray = edge_features_ndarray
        self.edges_src_ndarray = edges_src_ndarray
        self.edges_dst_ndarray = edges_dst_ndarray

        # print(node_labels_ndarray)
        self.num_classes = 4

        super().__init__(name='facebook')

    def process(self):
        node_features = torch.from_numpy(self.node_features_ndarray)
        node_labels = torch.from_numpy(self.node_labels_ndarray)
        edge_features = torch.from_numpy(self.edge_features_ndarray)
        edges_src = torch.from_numpy(self.edges_src_ndarray)
        edges_dst = torch.from_numpy(self.edges_dst_ndarray)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.N_nodes)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        n_nodes = self.N_nodes
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def list2array(node_labels_list, edge_list, node_features_list):
    N_nodes = len(node_labels_list) # the number of nodes
    N_edges = len(edge_list) # the number of edges
    # print(N_nodes)
    # node_indexes = np.array(node_labels_list).T[0].reshape((N_nodes, -1))
    # node_indexes = np.zeros((N_nodes, 1), dtype=np.float32)

    max_len_feature = 0
    for _node_feature in node_features_list:
        len_node_feature = len(_node_feature)
        if len_node_feature > max_len_feature: max_len_feature = len_node_feature

    node_features_ndarray = np.zeros((N_nodes, max_len_feature), dtype=np.int64)
    # print(node_features_ndarray.shape)
    # print(len(node_features_list))
    for i in range(len(node_features_list)):
        node_feature = node_features_list[i]
        for j in range(len(node_feature)):
            node_features_ndarray[i][j] = node_feature[j]

    # print(node_labels_list)
    # print(edge_list)
    # node_features_ndarray = np.array(node_features_list, dtype=np.int64)
    _node_labels = np.array(node_labels_list)
    # print(node_labels_list)
    node_labels_ndarray = np.array(node_labels_list, dtype=np.int64).T[1]
    edge_features_ndarray = np.ones(N_edges)
    edges_src_ndarray = np.array(edge_list).T[0]
    edges_dst_ndarray = np.array(edge_list).T[1]

    # print(edges_src_ndarray)

    return N_nodes, N_edges, node_features_ndarray, node_labels_ndarray, edge_features_ndarray, edges_src_ndarray, edges_dst_ndarray

if __name__ == '__main__':

    path_edge = './facebook_large/musae_facebook_edges.csv'
    path_node_labels = './facebook_large/musae_facebook_target.csv'
    path_node_features = './facebook_large/musae_facebook_features.json'
    _edge_list, node_labels_list, node_features_list = load_data(path_edge=path_edge, path_node_labels=path_node_labels, path_node_features = path_node_features)
    edge_list = []
    for edge in _edge_list:
        edge_list.append(edge)
        edge_list.append([edge[1], edge[0]])

    N_nodes, N_edges, node_features_ndarray, node_labels_ndarray, edge_features_ndarray, edges_src_ndarray, edges_dst_ndarray = list2array(node_labels_list=node_labels_list, edge_list=edge_list, node_features_list=node_features_list)
    dataset = FacebookDataset(N_nodes=N_nodes, N_edges=N_edges, node_features_ndarray=node_features_ndarray, node_labels_ndarray=node_labels_ndarray, edge_features_ndarray=edge_features_ndarray, edges_src_ndarray=edges_src_ndarray, edges_dst_ndarray=edges_dst_ndarray)
    graph = dataset[0]

    print(graph.ndata['feat'])


