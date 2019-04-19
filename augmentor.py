

import operator, msgpack, nltk, math, sys, os
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import dateutil.parser

from utils import settings, liwc_keys, lsm_keys, open_for_write, DEFAULT_TIMEZONE
from normalizer import expand_text, norms
#from contractions import contractions
from mention_graph import read_mention_names

liwc_words = {}
liwc_stems = {}
with open(settings['LIWC_PATH']) as handle:
    lines = handle.readlines()
    longest_stem = 0
    shortest_stem = 99
    for line in lines:
        tline = [part.strip() for part in line.strip().split(",")]
        if tline[0].endswith("*") and len(tline[0])-1 > longest_stem:
            longest_stem = len(tline[0])-1
        elif tline[0].endswith("*") and len(tline[0])-1 < shortest_stem:
            shortest_stem = len(tline[0])-1
    for i in range(shortest_stem, longest_stem+1):
        liwc_stems[i] = {}

    #liwc_keys = ["POSEMO"]#, "NEGEMO"]
    for line in lines:
        tline = [part.strip() for part in line.strip().split(",")]
        if tline[1] in liwc_keys:
            if tline[0].endswith("*"):
                tpat = tline[0][:-1]
                if tpat not in liwc_stems[len(tpat)]:
                    liwc_stems[len(tpat)][tpat] = [0]*len(liwc_keys)
                liwc_stems[len(tpat)][tpat][liwc_keys.index(tline[1])] = 1
            else:
                if tline[0] not in liwc_words:
                    liwc_words[tline[0]] = [0]*len(liwc_keys)
                liwc_words[tline[0]][liwc_keys.index(tline[1])] = 1

#print("LIWC keys: "+ str(liwc_keys))
print("LIWC words: " + str(len(liwc_words)))
print("LIWC set words: " + str(len(set(liwc_words))))
#print("liwcwords: " + str([key for key in list(liwc_words.keys())[:10]]))

print("LIWC stems: " + str(len(liwc_stems)))
print("LIWC set stems: " + str(len(set(liwc_stems))))
#print("liwcstems: " + str([key for key in list(liwc_stems[3].keys())[:10]]))
print("Stem range: " + str(shortest_stem) + "-" + str(longest_stem))
print("-"*20)

stop_words = set(stopwords.words('english'))

merge_dict = {}
with open('merged_utterances') as handle:
    for line in handle.readlines():
        parts = line.strip().split(':::')
        merge_dict[parts[0]] = parts[1]

def get_liwc_parts(token_set):
    rep_vec = np.array([0]*len(liwc_keys))
    match_words = [set() for i in range(len(liwc_keys))]
    for token in token_set:
        if token in liwc_words:
            rep_vec += np.array(liwc_words[token])
            for ind in np.array(liwc_words[token]).nonzero()[0]:
                match_words[ind].add(token)
        else:
            upper_lim = min(longest_stem+1, len(token))
            for i in range(shortest_stem, upper_lim):
                if token[:i] in liwc_stems[i]:
                    rep_vec += np.array(liwc_stems[i][token[:i]])
                    for ind in np.array(liwc_stems[i][token[:i]]).nonzero()[0]:
                        match_words[ind].add(token)
    return rep_vec.tolist(), [list(t_mw) for t_mw in match_words]

def main():
    er_file = open("error_files", "w")
    parser = ArgumentParser()
    parser.add_argument("-liwc", "--liwc-augment", dest="liwc", help="Augment the data with LIWC categories", default=False, action="store_true")
    parser.add_argument("-time", "--time", dest="time", help="Augment the data with time information", default=False, action="store_true")
    parser.add_argument("-normalize", "--normalize", dest="normalize", help="Augment the data with normalized text", default=False, action="store_true")
    parser.add_argument("-merge", "--merge", dest="merged", help="Augment the data with merged utterances for top utterances from file", default=False, action="store_true")
    parser.add_argument("-style", "--style", dest="style", help="Augment the data with style matching (LSM)", default=False, action="store_true")
    parser.add_argument("-freq", "--frequency", dest="freq", help="Augment the data with communication frequency information", default=False, action="store_true")
    parser.add_argument("-stopwords", "--stopword-augment", dest="stopwords", help="Augment the data with stopwords", default=False, action="store_true")
    parser.add_argument("-all", "--all", dest="all", help="Augment with all flags set to true", default=False, action="store_true")
    parser.add_argument("-mentions", "--mentions", dest="mentions", help="Augment with mentions of other tagged people", default=False, action="store_true")
    parser.add_argument("-uid", "--uid", dest="uid", help="Augment with unique message IDs accross entire corpus.", default=False, action="store_true")
    parser.add_argument("-context", "--context", dest="context", help="Number of context utterances to use for LSM (default 100)", default=100, type=int)
    parser.add_argument("-quick", "--quick", dest="quick", help="Augment more quickly by skipping the normalization and merging of utterances", default=False, action="store_true")
    opt = parser.parse_args()

    if opt.all:
        opt.liwc, opt.time, opt.style, opt.freq, opt.stopwords, opt.mentions, opt.uid, opt.normalize, opt.merged = [True]*9

    MSG_UNIQID = 0
    mention_names, first_names = [None]*2
    if opt.mentions:
        mention_names, first_names = read_mention_names()

    aug_set = [lp[1] for lp in [[opt.liwc, "LIWC"], [opt.stopwords, "stopwords"], [opt.time, "time"], [opt.freq, "frequency"], [opt.style, "style"], [opt.mentions, "mentions"], [opt.uid, "unique ID"], [opt.normalize, 'normalized'], [opt.merged, 'merged']] if lp[0]]
    if len(aug_set) == 0:
        print("You did not select any attributes to augment. Check possible flags with --help. Exiting...")
        sys.exit(0)

    print("Augmenting conversation files with " + str(aug_set) + "...")
    count = 1
    list_dir_set = os.listdir(settings['DATA_MERGE_DIR'])
    for filename in list_dir_set:
        tprt_str = "File (" + str(count) + "/" + str(len(list_dir_set)) + "): " + filename
        print("Merged " + tprt_str)
        convo = None
        with open(settings['DATA_MERGE_DIR'] + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()

        last_speaker = None
        last_time = None
        msg_counter = 0
        date_q = []

        # for LSM
        pmst = []
        liwc_sum_me = np.array([0]*len(liwc_keys))
        liwc_sum_other = np.array([0]*len(liwc_keys))
        total_me = 0
        total_other = 0
        num_utts = 0

        #### error list
        ctx_for_err = []

        for message in tqdm(convo[b"messages"]):
            pmst.append(message)
            prev_set = pmst[-opt.context:]
            num_utts += 1

            if b"text" not in message:
                continue
            msg_text = message[b"text"]
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()

            # add message date count
            mdate = dateutil.parser.parse(message[b"date"])
            if mdate.tzinfo == None:
                mdate = DEFAULT_TIMEZONE.localize(mdate)
            td = (mdate - last_time).seconds + (mdate - last_time).days*24*60*60 if last_time != None else 0
            current_speaker = message[b"user"].decode()

            if not opt.quick:
                # normalize the message
                if opt.normalize:
                    if b"normalized" not in message:
                        message[b"normalized"] = expand_text(norms, msg_text)
                if opt.merged:
                    merge_str = message[b"normalized"]
                    merge_str = merge_str.decode() if type(merge_str) == bytes else merge_str
                    message[b"merged"] = merge_utt(merge_str.strip())

            #print(msg_text)
            tokens = [_t for _t in nltk.word_tokenize(msg_text)]
            if opt.liwc:
                t_liwc_parts = get_liwc_parts(tokens)
                message[b"liwc_counts"] = t_liwc_parts[0]
                message[b"liwc_words"] = t_liwc_parts[1]
                message[b"words"] = len(tokens)

            if opt.stopwords:
                message[b"stopword_count"] = sum([1 if _t.lower() in stop_words else 0 for _t in tokens])

            #print(message[b'user'].decode() + ' (' + message[b'date'].decode() + '): ' + msg_text)
            ctx_for_err.append(message[b'user'].decode() + ' (' + message[b'date'].decode() + '): ' + msg_text)
            ctx_for_err = ctx_for_err[-5:]
            if opt.time:
                message[b"turn_change"] = last_speaker != current_speaker
                if td < 0: #assert td >= 0
                    er_file.write("\n.\n.\n.\n" + "\n".join(ctx_for_err))
                message[b"response_time"] = td
                if last_speaker != current_speaker:
                    last_time = mdate

            if opt.style:
                if len(pmst) > opt.context:
                    t_msg_text = prev_set[-opt.context][b"text"].decode()
                    t_tokens = [_t for _t in nltk.word_tokenize(t_msg_text)]
                    num_utts -= 1
                    if prev_set[-opt.context][b"user"].decode() in settings['my_name']:
                        liwc_sum_me -= prev_set[-opt.context][b"liwc_counts"]
                        total_me -= len(t_tokens)
                    else:
                        liwc_sum_other -= prev_set[-opt.context][b"liwc_counts"]
                        total_other -= len(t_tokens)
                    assert num_utts == opt.context and num_utts == len(prev_set)

                if message[b"user"].decode() in settings['my_name']:
                    liwc_sum_me += message[b"liwc_counts"]
                    total_me += len(tokens)
                else:
                    liwc_sum_other += message[b"liwc_counts"]
                    total_other += len(tokens)

                lsm_full = 0
                if msg_counter > 0:
                    lsm_vec_me = [liwc_sum_me[lv]*1.0/total_me if total_me > 0 else 0.0 for lv in range(len(liwc_keys)) if liwc_keys[lv] in lsm_keys]
                    lsm_vec_other = [liwc_sum_other[lv]*1.0/total_other if total_other > 0 else 0.0 for lv in range(len(liwc_keys)) if liwc_keys[lv] in lsm_keys]
                    lsm_full = np.array([0.0]*len(lsm_keys))
                    for lsm_ind in range(len(lsm_keys)):
                        lsm_full[lsm_ind] = 1.0 - abs(lsm_vec_me[lsm_ind] - lsm_vec_other[lsm_ind]) / (lsm_vec_other[lsm_ind] + lsm_vec_me[lsm_ind] + 0.0001)
                        #print(lsm_keys[lsm_ind] + ": " + str(lsm_full[lsm_ind]))
                    #print("\n\n")
                    lsm_full = np.average(lsm_full)
                message[b"lsm"] = lsm_full

            if opt.freq:
                message[b"all_freq"] = msg_counter
                m_mfreq, m_wfreq, m_dfreq = [0]*3
                new_dq = []
                for t_date in date_q:
                    td_t = mdate - t_date
                    #print(td_t)
                    if td_t.days < 30:
                        new_dq.append(t_date)
                        m_mfreq += 1
                    if td_t.days < 1:
                        m_dfreq += 1
                    if td_t.days < 7:
                        m_wfreq += 1

                date_q = new_dq

                message[b"month_freq"] = m_mfreq
                message[b"week_freq"] = m_wfreq
                message[b"day_freq"] = m_dfreq

            if opt.mentions:
                message[b"mentions"] = [0]*len(mention_names)
                for token in tokens:
                    if token in first_names:
                        if first_names[token] != cname:
                            message[b"mentions"][mention_names.index(first_names[token])] += 1

            if opt.uid:
                message[b'id'] = MSG_UNIQID
                MSG_UNIQID += 1

            # end of message loop
            msg_counter += 1
            last_speaker = current_speaker
            date_q.append(mdate)

        handle = open_for_write(settings['DATA_MERGE_DIR'] + "/" + filename, binary=True)
        handle.write(msgpack.packb(convo))
        handle.close()
        count += 1
    er_file.close()

def merge_utt(msg_text):
    ret_val = msg_text
    if msg_text in merge_dict:
        ret_val = merge_dict[msg_text]
    return ret_val

if __name__ == "__main__":
    main()
