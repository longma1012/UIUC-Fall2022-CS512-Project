# ------load_data, created and maintained by longma2------
import csv
import json
import random

def read_csv(path):
    output = []
    with open(path,"rt", newline='',encoding='latin-1') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        # print(data)
        line_count = 0
        for row in data:
            if line_count == 0:
                print('columns are:',row)
                line_count += 1
            else:
                # print(row)
                output.append(row)
    csvfile.close()
    return output

def read_json(path):
    output = []
    with open(path, newline='') as jsonfile:
        data = json.load(jsonfile)
        # print(data['0'])
        for i in range(len(data)):
            output.append(data[str(i)])
    jsonfile.close()
    return output

def label_organize(node_labels_list):
    output = []
    labels = ['government', 'politician', 'tvshow', 'company']
    for line in node_labels_list:
        node = int(line[0].split(',')[0])
        node_label = line[-1]
        node_label_num = None
        if node_label == '':
             node_label = line[-2]
        if node_label == '':
             node_label = line[-3]
        for label in labels:
            if label in node_label:
                node_label_num = labels.index(label)
        if node_label_num == None:
            print("problem line:", line)
        else:
            output.append([node,node_label_num])
    return output

def node_compare(node_labels_list, node_features_list):
    for i in range(len(node_features_list)):
        if i != node_labels_list[i][0]:
            random_label = random.randint(0,3)
            node_labels_list.insert(i,[i,random_label])
    return node_labels_list


def transfer2int(edge_list):
    res = []
    for i in range(len(edge_list)):
        pair = [int(edge_list[i][0]), int(edge_list[i][0])]
        res.append(pair)
    return res

def load_data(path_edge, path_node_labels, path_node_features):

    _edge_list = read_csv(path_edge)
    edge_list = transfer2int(_edge_list)
    node_labels_list = read_csv(path_node_labels)
    node_features_list = read_json(path_node_features)

    node_labels_list = label_organize(node_labels_list)
    node_labels_list = node_compare(node_labels_list, node_features_list)

    # edge_list = [[_[0]-1, _[1]-1] for _ in edge_list]
    # node_labels_list = [[_[0]-1, _[1]-1] for _ in node_labels_list]
    return edge_list, node_labels_list, node_features_list

if __name__ == '__main__':

    path_edge = './facebook_large/musae_facebook_edges.csv'
    path_node_labels = './facebook_large/musae_facebook_target.csv'
    path_node_features = './facebook_large/musae_facebook_features.json'
    edge_list, node_labels_list, node_features_list = load_data(path_edge=path_edge, path_node_labels=path_node_labels, path_node_features = path_node_features)
    # print(edge_list)
    print(node_labels_list)
    # print(node_features_list)
