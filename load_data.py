# ------load_data, created and maintained by longma2------

def read_citeseer(path, output):
    file = open(path, encoding='utf-8')
    for line in file.read().splitlines():
        line.replace("\n", "")
        _list = line.split(',')
        __list = [int(_) for _ in _list]
        output.append(__list[0:2])
    file.close()
    return

def load_data(path_edge, path_node_labels):
    edge_list = []
    node_labels_list = []

    read_citeseer(path=path_edge, output=edge_list)
    read_citeseer(path=path_node_labels, output=node_labels_list)

    edge_list = [[_[0]-1, _[1]-1] for _ in edge_list]
    node_labels_list = [[_[0]-1, _[1]-1] for _ in node_labels_list]
    return edge_list, node_labels_list

if __name__ == '__main__':

    path_edge = './citeseer_edges.txt'
    path_node_labels = './citeseer_node_labels.txt'
    edge_list, node_labels_list = load_data(path_edge=path_edge, path_node_labels=path_node_labels)
    # print(edge_list)
    print(node_labels_list)
