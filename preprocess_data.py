# ------pre-process_data, created and maintained by longma2------

import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
import os

from load_data import load_data




class CiteseerDataset(DGLDataset):
    def __init__(self, N_nodes, N_edges, node_indexes, node_labels_ndarray, edge_features_ndarray, edges_src_ndarray, edges_dst_ndarray):
        self.N_nodes = N_nodes
        self.N_edges = N_edges
        self.node_indexes = node_indexes
        self.node_labels_ndarray = node_labels_ndarray
        self.edge_features_ndarray = edge_features_ndarray
        self.edges_src_ndarray = edges_src_ndarray
        self.edges_dst_ndarray = edges_dst_ndarray

        # print(node_labels_ndarray)
        self.num_classes = 6

        super().__init__(name='citeseer')

    def process(self):
        node_features = torch.from_numpy(self.node_indexes) # citeseer dataset does not have node features, we decide to set node indexes as their features
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

def list2array(node_labels_list, edge_list):
    N_nodes = len(node_labels_list) # the number of nodes
    N_edges = len(edge_list) # the number of edges

    node_indexes = np.array(node_labels_list).T[0].reshape((N_nodes, -1))
    # node_indexes = np.zeros((N_nodes, 1), dtype=np.float32)
    node_labels_ndarray = np.array(node_labels_list, dtype=np.int64).T[1]
    edge_features_ndarray = np.ones(N_edges)
    edges_src_ndarray = np.array(edge_list).T[0]
    edges_dst_ndarray = np.array(edge_list).T[1]

    # print(edges_src_ndarray)

    return N_nodes, N_edges, node_indexes, node_labels_ndarray, edge_features_ndarray, edges_src_ndarray, edges_dst_ndarray

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

    print(graph.ndata['label'])
