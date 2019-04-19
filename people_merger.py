

import operator, msgpack, nltk, math, sys, os
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import dateutil.parser

from utils import settings, open_for_write, DEFAULT_TIMEZONE
from user_annotator import tag_set, read_people_file

# This file fixes timezone information and merges files from people with the same name.
# You have to run this to make the augmentor work and it makes the computations for LSM and frequency much easier.
# This file also adds the source type to the messages so that get_stats.py can still do everything it use to do.
def main():
    valid_people = read_people_file('tagged_people')
    persons = {name: [] for name in valid_people}

    print('Reading conversation files...')
    for filename in os.listdir(settings['DATA_DIR']):
        print("File: " + filename)
        convo = None
        ctype = filename.split("_")[0]
        with open(settings['DATA_DIR'] + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()
        if cname not in valid_people:
            print(cname + " is not in the list of tagged people...")
            continue

        for message in tqdm(convo[b"messages"]):
            if b"text" not in message:
                continue
            msg_text = message[b"text"]
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()
            message[b"type"] = ctype

            # add message date count
            mdate = dateutil.parser.parse(message[b"date"])
            if mdate.tzinfo == None:
                mdate = DEFAULT_TIMEZONE.localize(mdate)
            message[b"date"] = mdate
        persons[cname].append(convo[b'messages'])

    print('Merging people...')
    for name in tqdm(persons):
        msg_list = []
        prev_date = DEFAULT_TIMEZONE.localize(datetime.fromtimestamp(0))
        while len(persons[name]) > 0:
            mind, mval = [0, 99999999999999999]
            for q in range(0, len(persons[name])):
                td = persons[name][q][0][b'date'] - prev_date
                td = td.days*24*60*60 + td.seconds
                if td < mval:
                    mval = td
                    mind = q
            t_msg = persons[name][mind].pop(0)
            msg_list.append(t_msg)
            prev_date = t_msg[b'date']
            if len(persons[name][mind]) == 0:
                persons[name].remove([])

        handle = open_for_write(settings['DATA_MERGE_DIR'] + "/" + "_".join(name.split()), binary=True)
        convo = {'with': name, 'messages': msg_list}
        for msg in convo['messages']:
            msg[b'date'] = msg[b'date'].isoformat()
        handle.write(msgpack.packb(convo))
        handle.close()

if __name__ == "__main__":
    main()
