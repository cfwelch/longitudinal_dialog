


import operator, msgpack, math, sys, os, re
import numpy as np

from tqdm import tqdm

from utils import DATA_MERGE_DIR, p_fix, tag_set, uax_names, uax_set, open_for_write, read_people_file

def main():
    valid_people = read_people_file("tagged_people")

    for attribute in tag_set:
        handles = {val: open_for_write('stats/' + p_fix + '/ethall/' + '_'.join(attribute.split()) + '_' + val) for val in tag_set[attribute]}

        print("Reading conversation files...")
        for filename in os.listdir(DATA_MERGE_DIR):
            print("File: " + filename)
            convo = None
            with open(DATA_MERGE_DIR + "/" + filename, "rb") as handle:
                convo = msgpack.unpackb(handle.read())
            cname = convo[b"with"].decode()

            this_handle = handles[valid_people[cname][attribute]]

            for message in tqdm(convo[b"messages"]):
                if b"text" not in message:
                    continue
                msg_text = message[b"text"]
                if type(msg_text) == bytes:
                    msg_text = msg_text.decode()
                msg_text = msg_text.lower()
                this_handle.write(msg_text + '\n')

    print('Done!')

if __name__ == "__main__":
    main()
