

import operator, msgpack, nltk, math, sys, os
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser

from collections import Counter

import numpy as np
import dateutil.parser

from utils import f_my_name, liwc_keys, lsm_keys, open_for_write, p_fix, DATA_MERGE_DIR, LIWC_PATH, DEFAULT_TIMEZONE
from user_annotator import read_people_file, tag_set

def main():
    valid_people = read_people_file("tagged_people")
    parser = ArgumentParser()
    parser.add_argument("-context", "--context", dest="context", help="Number of context utterances to use for LSM (default 100)", default=100, type=int)
    opt = parser.parse_args()

    ppl_lsm = {person: [] for person in valid_people}

    for filename in os.listdir(DATA_MERGE_DIR):
        convo = None
        with open(DATA_MERGE_DIR + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()

        pmst = []
        liwc_sum_me = np.array([0]*len(liwc_keys))
        liwc_sum_other = np.array([0]*len(liwc_keys))
        total_me = 0
        total_other = 0
        num_utts = 0
        me_tok_counts = {}
        other_tok_counts = {}

        print("Calculating LSM for " + cname + "...")
        for message in tqdm(convo[b"messages"]):
            #if b"text" not in message:
            #    continue
            msg_text = message[b"text"]
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()

            mdate = dateutil.parser.parse(message[b"date"])
            if mdate.tzinfo == None:
                mdate = DEFAULT_TIMEZONE.localize(mdate)
            message = [mdate, msg_text, message[b"liwc_counts"], message[b"user"].decode()]

            pmst.append(message)
            prev_set = pmst[-opt.context:]

            num_utts += 1
            if len(pmst) > opt.context:
                msg_text = prev_set[-opt.context][1]
                tokens = [_t for _t in nltk.word_tokenize(msg_text)]
                num_utts -= 1
                if prev_set[-opt.context][3] in f_my_name:
                    liwc_sum_me -= prev_set[-opt.context][2]
                    total_me -= len(tokens)
                #    for _tok in tokens:
                #        me_tok_counts[_tok] -= 1
                else:
                    liwc_sum_other -= prev_set[-opt.context][2]
                    total_other -= len(tokens)
                #    for _tok in tokens:
                #        other_tok_counts[_tok] -= 1
                assert num_utts == opt.context and num_utts == len(prev_set)

            msg_text = message[1]
            tokens = [_t for _t in nltk.word_tokenize(msg_text)]

            if message[3] in f_my_name:
                liwc_sum_me += message[2]
                #for _tok in tokens:
                #    if _tok not in me_tok_counts:
                #        me_tok_counts[_tok] = 0
                #    me_tok_counts[_tok] += 1
                #total_me = sum([me_tok_counts[_z] for _z in me_tok_counts if me_tok_counts[_z] > 1])
                total_me += len(tokens)#sum(message[2])
            else:
                liwc_sum_other += message[2]
                #for _tok in tokens:
                #    if _tok not in other_tok_counts:
                #        other_tok_counts[_tok] = 0
                #    other_tok_counts[_tok] += 1
                #total_other = sum([other_tok_counts[_z] for _z in other_tok_counts if other_tok_counts[_z] > 1])
                total_other += len(tokens)#sum(message[2])

            if len(pmst) > opt.context:
                lsm_vec_me = [liwc_sum_me[lv]*1.0/total_me if total_me > 0 else 0.0 for lv in range(len(liwc_keys)) if liwc_keys[lv] in lsm_keys]
                lsm_vec_other = [liwc_sum_other[lv]*1.0/total_other if total_other > 0 else 0.0 for lv in range(len(liwc_keys)) if liwc_keys[lv] in lsm_keys]
                lsm_full = np.array([0.0]*len(lsm_keys))
                for lsm_ind in range(len(lsm_keys)):
                    lsm_full[lsm_ind] = 1.0 - abs(lsm_vec_me[lsm_ind] - lsm_vec_other[lsm_ind]) / (lsm_vec_other[lsm_ind] + lsm_vec_me[lsm_ind] + 0.0001)
                    #print(lsm_keys[lsm_ind] + ": " + str(lsm_full[lsm_ind]))
                #print("\n\n")
                lsm_full = np.average(lsm_full)
                ppl_lsm[cname].append(lsm_full)

    handle = open_for_write('stats/' + p_fix + '/lsm_over_messages.csv')
    print("Writing LSM to CSV...")
    smooth = 100
    ppl_lsm_smooth = {person: None for person in valid_people}
    for person in ppl_lsm:
        #if len(ppl_lsm[person]) < 20*smooth:
        #    continue
        handle.write(person)
        counter = 0
        lsm_set = [0]
        for k in range(len(ppl_lsm[person])):
            lsm_set[-1] += ppl_lsm[person][k]
            counter += 1
            if counter == smooth:# smooth over this num
                lsm_set.append(0)# lsm_set[-1] I think you need to append 0 if you are not doing cumulative
                counter = 0
        divisor = 1
        for k in lsm_set[:-1]:
            handle.write('\t' + str(k*1.0/smooth))# over divisor if cumulative
            divisor += 1
        ppl_lsm_smooth[person] = lsm_set[:-1]
        handle.write('\n')
    handle.close()

    max_blen = max([len(ppl_lsm_smooth[person]) for person in ppl_lsm_smooth])

    handle = open_for_write('stats/' + p_fix + '/lsm_groups_over_messages.csv')
    print('Writing LSM tags to CSV...')
    for tag in tag_set:
        handle.write(tag)
        ppl_grp_lsm = {tval: {i: [] for i in range(max_blen)} for tval in tag_set[tag]}
        for person in ppl_lsm_smooth:
            for k in range(len(ppl_lsm_smooth[person])):
                ppl_grp_lsm[valid_people[person][tag]][k].append(ppl_lsm_smooth[person][k])
        for tval in tag_set[tag]:
            handle.write('\n' + tval + '\taverage LSM')
            for k in range(max_blen):
                #print("datblen: " + str(ppl_grp_lsm[tval][k]))
                avg_over_ppl = sum(ppl_grp_lsm[tval][k])*1.0/len(ppl_grp_lsm[tval][k])/(k+1) if len(ppl_grp_lsm[tval][k]) else 0
                handle.write('\t' + str(avg_over_ppl))
            handle.write('\n' + tval + '\tpeople')
            for k in range(max_blen):
                handle.write('\t' + str(len(ppl_grp_lsm[tval][k])))
        handle.write('\n\n')

if __name__ == "__main__":
    main()
