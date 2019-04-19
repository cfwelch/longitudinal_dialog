

import random, msgpack, math, sys, os, re

from utils import p_fix, tag_set
from mention_graph import read_mention_names

def main():
    ppl_map, first_names = read_mention_names()

    inst_id = input('Choose a person ID: ')
    inst_id = int(inst_id)
    print('You chose: ' + ppl_map[inst_id])

    instances = read_split_files()
    print('Loaded ' + str(len(instances)) + ' dialog contexts!')
    p_spe = [inst for inst in instances if inst[-1][10] == inst_id]
    print('There are ' + str(len(p_spe)) + ' instances for this person.')

    for ctx in p_spe:
        for i in range(0, len(ctx)):
            if ctx[i] == None:
                continue
            print('Utterance ' + str(i+1) + ' @ (' + str(ctx[i][0].decode()) + ') ' + str(ctx[i][1].decode()) + ': ' + str(ctx[i][2].decode()))
        input()

def read_split_files():
    instances = []
    for ctx_link in ['train_contexts', 'dev_contexts', 'test_contexts']:
        with open('splits/' + p_fix + '/train_contexts', 'rb') as handle:
            objs = msgpack.unpackb(handle.read())
            for o_ in objs:
                instances.append(o_)
    return instances

if __name__ == "__main__":
    main()
