

import operator, msgpack, nltk, math, sys, os
from prettytable import PrettyTable
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm

import numpy as np
import dateutil.parser

from utils import f_my_name, liwc_keys, lsm_keys, p_fix, open_for_write, START_TIME, DEFAULT_TIMEZONE
from user_annotator import read_people_file, tag_set

CONVO_DIR = "conversations/" + p_fix

token_me = {}
token_other = {}
stop_words = set(stopwords.words('english'))
TOP_UTT = 25
my_utts = {}
other_utts = {}

# month bin should be a factor of 12
month_bin = 1
# date ranges to cover
months = ["0" + str(a) if len(str(a)) < 2 else str(a) for a in range(1, 13, month_bin)]
years = [str(y) for y in range(START_TIME.year, datetime.now().year+1)]

# convo bins
cbins = [100, 999999]

def main():
    valid_people = read_people_file("tagged_people")

    people = {f_my_name: 0}
    empty_messages = 0
    empties = []
    begins_me = {}
    begins_other = {}
    lengths = []

    bmc_me = {x:{} for x in cbins}
    bmc_other = {x:{} for x in cbins}
    ppl_convos = {}

    print("Reading conversation files...")
    for filename in tqdm(os.listdir(CONVO_DIR)):
        #print("File: " + filename)
        convo = None
        with open(CONVO_DIR + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()
        start_date = dateutil.parser.parse(convo[b'start']).replace(tzinfo=DEFAULT_TIMEZONE)
        if cname not in valid_people:
            #print(cname + " is not in the list of tagged people...")
            continue
        if cname not in people:
            people[cname] = 0
        people[cname] += 1
        if cname not in ppl_convos:
            ppl_convos[cname] = {}
        ppl_convos[cname][start_date] = convo

    ppl_lsm = {person: [] for person in ppl_convos}
    print("Looping over people...")
    for person in tqdm(ppl_convos):
        nth_convo = 0
        for k,convo in sorted(ppl_convos[person].items()):
            #print(person + ": date key is: " + str(k) + " -- convo " + str(nth_convo))

            msgcount = 0
            lengths.append(len(convo[b'messages']))
            liwc_sum_me = np.array([0]*len(liwc_keys))
            liwc_sum_other = np.array([0]*len(liwc_keys))
            total_me = 0
            total_other = 0
            for message in convo[b"messages"]:
                if str(message[b"date"]) < str(START_TIME):
                    continue
                if b"text" not in message:
                    empty_messages += 1
                    continue
                msg_text = message[b"text"]
                if type(msg_text) == bytes:
                    msg_text = msg_text.decode()
                msg_text = msg_text.lower()

                # add message date count
                mdate = dateutil.parser.parse(message[b"date"])
                mmstr = str(((mdate.month-1) // month_bin) * month_bin + 1)
                mdate = str(mdate.year) + ":" + ("0" + mmstr if len(mmstr) < 2 else mmstr)

                # add up tokens and number of incoming and outgoing messages
                tokens = [_t for _t in nltk.word_tokenize(msg_text)]

                if message[b"user"].decode() in f_my_name:
                    liwc_sum_me += message[b'liwc_counts']
                    total_me += len(tokens)
                else:
                    liwc_sum_other += message[b'liwc_counts']
                    total_other += len(tokens)

                if msgcount == len(convo[b"messages"])-1:#0 for begin, len(convo[b"messages"])-1 for end
                    bmc_index = cbins[len(cbins) - sum([1 for _i in cbins if nth_convo < _i])]

                    if message[b"user"].decode() in f_my_name:# if the person is me
                        if msg_text not in begins_me:
                            begins_me[msg_text] = 0
                        begins_me[msg_text] += 1

                        if msg_text not in bmc_me[bmc_index]:
                            bmc_me[bmc_index][msg_text] = 0
                        bmc_me[bmc_index][msg_text] += 1

                        for token in tokens:
                            if token in stop_words:
                                continue
                            if token not in token_me:
                                token_me[token] = 0
                            token_me[token] += 1
                    else:
                        if msg_text not in begins_other:
                            begins_other[msg_text] = 0
                        begins_other[msg_text] += 1

                        if msg_text not in bmc_other[bmc_index]:
                            bmc_other[bmc_index][msg_text] = 0
                        bmc_other[bmc_index][msg_text] += 1

                        for token in tokens:
                            if token in stop_words:
                                continue
                            if token not in token_other:
                                token_other[token] = 0
                            token_other[token] += 1

                msgcount += 1
            nth_convo += 1
            if len(convo[b"messages"]) > 10:
                lsm_vec_me = [liwc_sum_me[lv]*1.0/total_me if total_me > 0 else 0.0 for lv in range(len(liwc_keys)) if liwc_keys[lv] in lsm_keys]
                lsm_vec_other = [liwc_sum_other[lv]*1.0/total_other if total_other > 0 else 0.0 for lv in range(len(liwc_keys)) if liwc_keys[lv] in lsm_keys]
                lsm_full = np.array([0.0]*len(lsm_keys))
                for lsm_ind in range(len(lsm_keys)):
                    lsm_full[lsm_ind] = 1.0 - abs(lsm_vec_me[lsm_ind] - lsm_vec_other[lsm_ind]) / (lsm_vec_other[lsm_ind] + lsm_vec_me[lsm_ind] + 0.0001)
                    #print(lsm_keys[lsm_ind] + ": " + str(lsm_full[lsm_ind]))
                #print("\n\n")
                lsm_full = np.average(lsm_full)
                #print("Full LSM: " + str(lsm_full))
                ppl_lsm[person].append(lsm_full)

    handle = open_for_write('stats/' + p_fix + '/lsm_over_convo.csv')
    for person in tqdm(ppl_lsm):
        handle.write(person)
        counter = 0
        lsm_set = [0]
        for k in range(len(ppl_lsm[person])):
            lsm_set[-1] += ppl_lsm[person][k]
            counter += 1
            if counter == 1:# smooth over this num
                lsm_set.append(0)
                counter = 0
        for k in lsm_set:
            handle.write('\t' + str(k))
        handle.write('\n')
    handle.close()

    print("\nTop tokens from me: " + str(get_top_words(token_me)))
    print("\nTop tokens from others: " + str(get_top_words(token_other)))

    print("\nBeginnings of conversations from me: ")
    for k,v in sorted(begins_me.items(), key=operator.itemgetter(1), reverse=True)[:TOP_UTT]:
        print("[" + str(v) + "]: " + k)
    print("\nBeginnings of conversations from other: ")
    for k,v in sorted(begins_other.items(), key=operator.itemgetter(1), reverse=True)[:TOP_UTT]:
        print("[" + str(v) + "]: " + k)

    #print("\nBeginnings of conversations from me per bin: ")
    #for cbin in cbins:
    #    print("Bin size is: " + str(cbin))
    #    for k,v in sorted(bmc_me[cbin].items(), key=operator.itemgetter(1), reverse=True)[:TOP_UTT]:
    #        print("[" + str(v) + "]: " + k)

    print("\nDistribution of conversations by person: ")
    for k,v in sorted(people.items(), key=operator.itemgetter(1), reverse=True):
        print("[" + str(v) + "]: " + k)

    print("\nNumber of conversations: " + str(len(lengths)))
    print("Max conversation length: " + str(max(lengths)))
    print("Min conversation length: " + str(min(lengths)))
    print("Average conversation length: " + str(np.average(lengths)))
    print("Standard deviation length: " + str(np.std(lengths)))

    for i in range(1, 11):
        print("Number of length " + str(i) + "s: " + str(lengths.count(i)))

def get_top_words(token_set, TOP_WORDS=50):
    top_tokens = []
    for k,v in sorted(token_set.items(), key=operator.itemgetter(1), reverse=True)[:TOP_WORDS]:
        top_tokens.append(k)
    return top_tokens

if __name__ == "__main__":
    main()
