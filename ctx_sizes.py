

import msgpack
from collections import defaultdict

def main():
    tlabel_dict = defaultdict(lambda: 0)

    label_dict = defaultdict(lambda: 0)
    with open('dev_contexts', 'rb') as handle:
        ctxs = msgpack.unpackb(handle.read())
    for ctx in ctxs:
        label_dict[ctx[b'label']] += 1
        tlabel_dict[ctx[b'label']] += 1
    dlen = len(ctxs)
    print(label_dict)
    print('\n')

    label_dict = defaultdict(lambda: 0)
    with open('train_contexts', 'rb') as handle:
        ctxs = msgpack.unpackb(handle.read())
    for ctx in ctxs:
        label_dict[ctx[b'label']] += 1
        tlabel_dict[ctx[b'label']] += 1
    tlen = len(ctxs)
    print(label_dict)
    print('\n')

    label_dict = defaultdict(lambda: 0)
    with open('test_contexts', 'rb') as handle:
        ctxs = msgpack.unpackb(handle.read())
    for ctx in ctxs:
        label_dict[ctx[b'label']] += 1
        tlabel_dict[ctx[b'label']] += 1
    elen = len(ctxs)
    print(label_dict)
    print('\n')

    print('Train length is: ' + str(tlen))
    print('Development length is: ' + str(dlen))
    print('Test length is: ' + str(elen))
    print('Total length is: ' + str(elen+tlen+dlen))
    print(label_dict)

if __name__ == "__main__":
    main()
