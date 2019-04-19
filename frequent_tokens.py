

import msgpack, operator, nltk, math, json, re, os

from collections import Counter
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words, wordnet
from pprint import pprint
from datetime import datetime
from contractions import contractions

import dateutil.parser
import numpy as np

from utils import DATA_DIR, f_my_name, EMBEDDING_LOCATION, EMBEDDING_SIZE
from user_annotator import tag_set, read_people_file

TOP_WORDS = 150

#print(type(wordnet.keys()))
manywords = set(words.words())
wnl = WordNetLemmatizer()
norms = {}

def main():
    valid_people = read_people_file("tagged_people")
    # incoming means to me, outgoing means from me
    people_group = {inout: {tag: {tval: [] for tval in tag_set[tag]} for tag in tag_set} for inout in ["incoming", "outgoing"]}

    unique_tokens = set()

    print("Reading conversation files...")
    for filename in os.listdir(DATA_DIR):
        print("File: " + filename)
        convo = None
        with open(DATA_DIR + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()
        if cname not in valid_people:
            print(cname + " is not in the list of tagged people...")
            continue

        #mdate = dateutil.parser.parse(message[b"date"])

        for message in tqdm(convo[b"messages"]):
            if b"text" not in message:
                continue
            msg_text = message[b"text"]
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()

            tokens = [_t.lower() for _t in nltk.word_tokenize(msg_text) if _t.isalpha()]

            for token in tokens:
                _t_ = wnl.lemmatize(token)
                unique_tokens.add(token)
                if _t_ not in manywords and token not in manywords:
                    ikey = "outgoing" if message[b"user"].decode() in f_my_name else "incoming"
                    for tag in tag_set:
                        people_group[ikey][tag][valid_people[cname][tag]].append(token)

    THRESHOLD = 5
    top_by_grp = {}
    ikey = "outgoing"
    for tag in tag_set:
        for t_val in tag_set[tag]:
            #for ikey in ["incoming", "outgoing"]:
            counts = Counter(people_group[ikey][tag][t_val])
            for k in list(counts):
                if counts[k] < THRESHOLD:
                    del counts[k]
            counts_sum = sum([_[1] for _ in counts.items()])
            c_norms = {_[0]: _[1]*1.0/counts_sum for _ in counts.items()}

            #for o_tag in tag_set:# add this outer loop to get words unique across all groups
            u_others = []
            for o_val in tag_set[tag]:#o_tag
                if o_val != t_val:# or o_tag != tag:
                    u_others += people_group[ikey][tag][o_val]#o_tag
            u_counts = Counter(u_others)
            for k in list(u_counts):
                if u_counts[k] < THRESHOLD:
                    del u_counts[k]
            u_sum = sum([_[1] for _ in u_counts.items()])
            u_norms = {_[0]: _[1]*1.0/u_sum for _ in u_counts.items()}
            for key in u_norms.keys():
                if key in c_norms:
                    c_norms[key] -= u_norms[key]

            top_c = [_[0] for _ in Counter(c_norms).most_common(TOP_WORDS)]#,_[1])
            top_by_grp[tag + "_" + t_val] = set(top_c)

    for tag in tag_set:
        print("="*30 + "\nTag: " + tag)
        for t_val in tag_set[tag]:
            union_set = set()
            for o_tag in tag_set:
                for o_val in tag_set[o_tag]:
                    if o_tag != tag or o_val != t_val:
                        union_set = union_set.union(top_by_grp[o_tag + "_" + o_val])
            #print('union set: ' + str(union_set))

            temp_set = set(top_by_grp[tag + "_" + t_val]).difference(union_set)
            print(t_val + "(set diff) - " + str(temp_set) + "\n")
            print(t_val + "(" + ikey + ") - " + str(top_by_grp[tag + "_" + t_val]) + "\n")

    #show_not_glove(unique_tokens)

def show_not_glove(unique_tokens):
    # Which words do not have embeddings in GloVe?
    gloves = []
    print('Loading pretrained embeddings...')
    with open(EMBEDDING_LOCATION + 'glove.840B.300d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = " ".join(values[:len(values)-EMBEDDING_SIZE])
            gloves.append(word)

    for token in unique_tokens:
        if token not in gloves:
            print(token)

if __name__ == "__main__":
    main()
