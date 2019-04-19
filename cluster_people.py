

import msgpack, os
import networkx as nx
import community as co
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from utils import DATA_MERGE_DIR, read_people_file, make_user_vector
from mention_graph import read_mention_names

def main():
    valid_people = read_people_file("tagged_people")
    # so you have a unique index per name
    ppl_map, first_names = read_mention_names()
    train_graph = np.zeros((len(ppl_map), len(ppl_map)))
    ppl_vecs = {name: 0 for name in ppl_map}

    print("Reading conversation files...")
    for filename in os.listdir(DATA_MERGE_DIR):
        print("File: " + filename)
        convo = None
        with open(DATA_MERGE_DIR + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()
        if cname not in valid_people:
            print(cname + " is not in the list of tagged people...")
            continue

        user_vec = make_user_vector(valid_people, cname)
        ppl_vecs[cname] = user_vec
        for message in tqdm(convo[b"messages"]):
            train_graph[ppl_map.index(cname)] += message[b'mentions']

    sample_graph = nx.Graph()
    #sample_graph.add_edge("a", "b", weight=1.)
    train_max = np.max(train_graph)
    train_min = np.min(train_graph)

    for t_gx in range(len(ppl_map)):
        for t_gy in range(len(ppl_map)):
            # higher weight should be shorter distance -- this uses distance
            train_graph[t_gx][t_gy] = 1 - (train_max - train_graph[t_gx][t_gy]) * 1.0 / (train_max - train_min)
            if train_graph[t_gx][t_gy] > 0.0:
                sample_graph.add_edge(t_gx, t_gy, weight=train_graph[t_gx][t_gy])
    #train_graph = floyd_warshall(train_graph)
    #train_graph = np.where(train_graph != np.inf, train_graph, 100.0)

    partition = co.best_partition(sample_graph)

    p = defaultdict(list)
    for node, com_id in partition.items():
        p[com_id].append(node)

    sum_vec = np.sum(np.array([ppl_vecs[name] for name in ppl_map]), axis=0)

    for com, nodes in p.items():
        print('\n' + '-'*30)
        their_names = [ppl_map[node] for node in nodes]
        sum_atts = np.sum(np.array([ppl_vecs[name] for name in their_names]), axis=0)
        sum_atts = sum_atts / sum_vec
        print(com, their_names)
        print('ids: ' + str(nodes))
        print('atts: ' + str(sum_atts))

if __name__ == "__main__":
    main()
