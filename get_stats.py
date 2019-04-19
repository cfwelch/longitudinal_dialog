

import operator, msgpack, nltk, math, sys, os

from prettytable import PrettyTable
from collections import defaultdict
from nltk.corpus import stopwords
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

import matplotlib.pyplot as plt
import dateutil.parser
import numpy as np

from utils import START_TIME, MSG_TYPE, settings, liwc_keys, open_for_write, read_people_file, tag_set
#from normalizer import expand_text, norms
#from contractions import contractions
from response_times import get_rt_stats
from aggregate_stats import get_agg_stats
from ttest_liwc import ttest_make_heat_maps
from mention_graph import make_mention_graph

token_me = {}
token_other = {}
stop_words = set(stopwords.words('english'))
my_utts = {}
other_utts = {}
other_utt_dict = defaultdict(lambda: defaultdict(lambda: 0))
my_utt_dict = defaultdict(lambda: 0)

# date ranges to cover
months = None
years = [str(y) for y in range(START_TIME.year, datetime.now().year+1)]

def main():
    global years
    parser = ArgumentParser()
    # month bin should be a factor of 12
    parser.add_argument("-m", "--month-bins", dest="month_bins", help="Number of months to bin together. Default is 6.", default=6, type=int) # 6 looks good in tikz
    parser.add_argument("-tu", "--top-utterances", dest="top_utt", help="Number of top utterances to show. Defaults to 100.", default=100, type=int)
    parser.add_argument("-mc", "--mention-cutoff", dest="mention_cutoff", help="Number of mentions an edge must have to be included in the mention graph. Defaults to 5.", default=5, type=int)
    parser.add_argument("-mp", "--mention-people", dest="mention_people", help="Number of people to include in the mention graph. Defaults to all.", default=-1, type=int)
    parser.add_argument("-gm", "--generate-main", dest="main", help="Generate main statistics for LIWC, groups, and counts. Defaults to False.", default=False, action="store_true")
    parser.add_argument("-gr", "--generate-response-times", dest="resptime", help="Generate response time statistics. Defaults to False.", default=False, action="store_true")
    parser.add_argument("-ga", "--generate-aggregate", dest="aggregate", help="Generate aggregate message statistics. Defaults to False.", default=False, action="store_true")
    parser.add_argument("-gt", "--generate-ttest-liwcmap", dest="sigliwc", help="Generate LIWC heatmaps. Defaults to False.", default=False, action="store_true")
    parser.add_argument("-gg", "--generate-mention-graph", dest="mentions", help="Generate mentions graph. Defaults to False.", default=False, action="store_true")
    parser.add_argument("-all", "--all", dest="all", help="Generate all statistics.", default=False, action="store_true")
    opt = parser.parse_args()

    if opt.all:
        opt.main, opt.resptime, opt.aggregate, opt.sigliwc, opt.mentions = [True]*5

    gen_opts = [opt.main, opt.resptime, opt.aggregate, opt.sigliwc, opt.mentions]
    if True not in gen_opts:
        print("See --help for options for generating statistics and select at least one when running this script.")
        sys.exit(0)

    if opt.main:
        print("="*30 + "\nGenerating Main Statistics\n" + "="*30)
        main_stats(opt.month_bins, opt.top_utt)
    if opt.resptime:
        print("="*30 + "\nGenerating Response Time Statistics\n" + "="*30)
        get_rt_stats()
    if opt.aggregate:
        print("="*30 + "\nGenerating Aggregate Statistics\n" + "="*30)
        get_agg_stats(years, opt.month_bins)
    if opt.sigliwc:
        print("="*30 + "\nGenerating LIWC Heatmaps\n" + "="*30)
        ttest_make_heat_maps(save_as_fig=True)
    if opt.mentions:
        print("="*30 + "\nGenerating Mentions Graph\n" + "="*30)
        make_mention_graph(opt.mention_cutoff, opt.mention_people)

def main_stats(month_bins, top_utt):
    global months, years
    months = ["0" + str(a) if len(str(a)) < 2 else str(a) for a in range(1, 13, month_bins)]

    valid_people = read_people_file("tagged_people")
    names = []#'Veronica Diebold'] # use this for testing on small sets

    people = {settings['my_name'][0]: [0]*len(MSG_TYPE)}
    people_parts = {settings['my_name'][0]: make_person()}
    people_group = {inout: {tag: {tval: make_person() for tval in tag_set[tag]} for tag in tag_set} for inout in ["incoming", "outgoing"]}
    empty_messages = 0
    total_words = 0
    total_words_me = 0
    total_words_other = 0
    empties = []

    print("Reading conversation files...")
    for filename in os.listdir(settings['DATA_MERGE_DIR']):
        print("File: " + filename)
        convo = None
        with open(settings['DATA_MERGE_DIR'] + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()
        if len(names) > 0 and cname not in names:
            continue
        if cname not in valid_people:
            print(cname + " is not in the list of tagged people...")
            continue
        if cname not in people:
            people[cname] = [0]*len(MSG_TYPE)
            people_parts[cname] = make_person()

        for message in tqdm(convo[b"messages"]):
            if str(message[b"date"]) < str(START_TIME):
                continue
            if b"text" not in message:
                empty_messages += 1
                continue
            msg_text = message[b"text"]
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()
            msg_text = msg_text.lower()
            # contractions are not one-way and are not really necessary
            #msg_text = expand_text(contractions, msg_text)
            #mb4 = " " + msg_text
            #msg_text = expand_text(norms, msg_text)

            #if msg_text == "":
            #    empties.append(mb4)

            # add message date count
            mdate = dateutil.parser.parse(message[b"date"])
            mmstr = str(((mdate.month-1) // month_bins) * month_bins + 1)
            mdate = str(mdate.year) + ":" + ("0" + mmstr if len(mmstr) < 2 else mmstr)

            # add up tokens and number of incoming and outgoing messages
            tokens = [_t for _t in nltk.word_tokenize(msg_text)]
            total_words += len(tokens)
            ctype = message[b"type"].decode()
            people[cname][MSG_TYPE[ctype]] += 1
            people[settings['my_name'][0]][MSG_TYPE[ctype]] += 1

            # Error if the messages do not have LIWC counts
            if b"liwc_counts" not in message:
                print("Error: LIWC counts not found in messages, run LIWC augmentor and try again...")
                sys.exit(1)
            elif b"response_time" not in message:
                print("Error: Response time not found in messages, run time augmentor and try again...")
                sys.exit(1)

            if message[b"user"].decode() in settings['my_name']:# if the person is me
                people_parts[settings['my_name'][0]]["liwc_counts"][mdate] += np.array(message[b"liwc_counts"])
                for ind in range(len(liwc_keys)):
                    people_parts[settings['my_name'][0]]["liwc_words"][ind].extend(message[b"liwc_words"][ind])
                people_parts[settings['my_name'][0]]["chrono"][mdate]["words"] += len(tokens)
                people_parts[settings['my_name'][0]]["chrono"][mdate]["messages"] += 1
                people_parts[cname]["outgoing"] += 1
                people_parts[cname]["out_words"] += len(tokens)

                for tag in tag_set:
                    people_group["outgoing"][tag][valid_people[cname][tag]]["liwc_counts"][mdate] += np.array(message[b"liwc_counts"])
                    for ind in range(len(liwc_keys)):
                        people_group["outgoing"][tag][valid_people[cname][tag]]["liwc_words"][ind].extend(message[b"liwc_words"][ind])
                    people_group["outgoing"][tag][valid_people[cname][tag]]["chrono"][mdate]["words"] += len(tokens)
                    people_group["outgoing"][tag][valid_people[cname][tag]]["chrono"][mdate]["messages"] += 1
                    if message[b"turn_change"]:
                        people_group["outgoing"][tag][valid_people[cname][tag]]["response_times"][mdate].append(message[b"response_time"])

                # count whole utterances
                if msg_text not in my_utts:
                    my_utts[msg_text] = 0
                my_utts[msg_text] += 1
                # quick processing will give empty strings
                nm_text = message[b'merged'].decode() if b'merged' in message else ''
                my_utt_dict[nm_text] += 1

                # back to counting tokens
                total_words_me += len(tokens)
                for token in tokens:
                    if token in stop_words:
                        continue
                    if token not in token_me:
                        token_me[token] = 0
                    token_me[token] += 1
            else:# count tokens from other person
                people_parts[cname]["liwc_counts"][mdate] += np.array(message[b"liwc_counts"])
                for ind in range(len(liwc_keys)):
                    people_parts[cname]["liwc_words"][ind].extend(message[b"liwc_words"][ind])
                people_parts[cname]["chrono"][mdate]["words"] += len(tokens)
                people_parts[cname]["chrono"][mdate]["messages"] += 1
                people_parts[cname]["incoming"] += 1
                people_parts[cname]["in_words"] += len(tokens)

                for tag in tag_set:
                    people_group["incoming"][tag][valid_people[cname][tag]]["liwc_counts"][mdate] += np.array(message[b"liwc_counts"])
                    for ind in range(len(liwc_keys)):
                        people_group["incoming"][tag][valid_people[cname][tag]]["liwc_words"][ind].extend(message[b"liwc_words"][ind])
                    people_group["incoming"][tag][valid_people[cname][tag]]["chrono"][mdate]["words"] += len(tokens)
                    people_group["incoming"][tag][valid_people[cname][tag]]["chrono"][mdate]["messages"] += 1
                    if message[b"turn_change"]:
                        people_group["incoming"][tag][valid_people[cname][tag]]["response_times"][mdate].append(message[b"response_time"])

                # count whole utterances
                if msg_text not in other_utts:
                    other_utts[msg_text] = 0
                other_utts[msg_text] += 1
                #other_utt_dict[cname][msg_text] += 1
                #normd_text = message[b"normalized"].decode()
                # quick processing will give empty strings
                nm_text = message[b'merged'].decode() if b'merged' in message else ''
                other_utt_dict[cname][nm_text] += 1

                # count tokens from other person
                total_words_other += len(tokens)
                for token in tokens:
                    if token in stop_words:
                        continue
                    if token not in token_other:
                        token_other[token] = 0
                    token_other[token] += 1
        #people_parts[cname]["messages"] = convo[b"messages"]

    #print(people_parts[name]["liwc"])
    #for l_cat in liwc_keys:
    #    for key in people_parts[name]["liwc"]:
    #        print(key + " - " + l_cat + ": " + str(people_parts[name]["liwc"][key][l_cat]))

    #print("EMPTIES: " + str(empties))
    total = sum([sum(v) for k,v in people.items() if k not in settings['my_name']])
    print("\nNumber of total messages: " + "{:,}".format(total))
    print("Number of total tokens: " + "{:,}".format(total_words_other + total_words_me))
    print("Number of total tokens from me: " + "{:,}".format(total_words_me))
    print("Number of total tokens from others: " + "{:,}".format(total_words_other))

    mt_set = set(token_me.keys())
    ot_set = set(token_other.keys())
    t_both = mt_set.union(ot_set)
    print("Number of unique tokens from me: " + "{:,}".format(len(mt_set)))
    print("Number of unique tokens from others: " + "{:,}".format(len(ot_set)))
    print("Number of total unique tokens: " + "{:,}".format(len(t_both)))
    print("Number of tokens only I use: " + "{:,}".format(len(mt_set-ot_set)))
    print("Number of tokens only others use: " + "{:,}".format(len(ot_set-mt_set)))

    total_me_outgoing = sum([people_parts[_z]["outgoing"] for _z in people_parts.keys()])
    mm_set = set(my_utts.keys())
    om_set = set(other_utts.keys())
    um_both = mm_set.union(om_set)
    print("\nNumber of empty messages: " + "{:,}".format(empty_messages))
    print("Number of total messages from me: " + "{:,}".format(total_me_outgoing))
    print("Number of unique messages from me: " + "{:,}".format(len(my_utts)))
    print("Number of unique messages from others: " + "{:,}".format(len(other_utts)))
    print("Number of total unique messages: " + "{:,}".format(len(um_both)))

    handle = open_for_write("stats/" + settings['prefix'] + "/total_uniques.txt")
    handle.write("tokens me: " + str(len(mt_set)) + "\n")
    handle.write("tokens others: " + str(len(ot_set)) + "\n")
    handle.write("tokens total: " + str(len(t_both)) + "\n")
    handle.write("messages me: " + str(len(my_utts)) + "\n")
    handle.write("messages others: " + str(len(other_utts)) + "\n")
    handle.write("messages total: " + str(len(um_both)) + "\n")
    handle.close()

    print("\nMy top " + str(top_utt) + " messages: ")
    handle = open_for_write("stats/" + settings['prefix'] + "/my_top_utterances")
    for k,v in sorted(my_utts.items(), key=operator.itemgetter(1), reverse=True)[:top_utt]:
        out_str = "[" + "{:,}".format(v) + "] \"" + k + "\""
        print(out_str)
        handle.write(out_str + '\n')
    handle.close()

    handle = open_for_write("stats/" + settings['prefix'] + "/my_top_merged")
    for k,v in sorted(my_utt_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_utt]:
        handle.write("[" + "{:,}".format(v) + "] \"" + k + "\"\n")
    handle.close()

    print("\nOther top " + str(top_utt) + " messages: ")
    for k,v in sorted(other_utts.items(), key=operator.itemgetter(1), reverse=True)[:top_utt]:
        print("[" + "{:,}".format(v) + "] \"" + k + "\"")

    # Write individual speaker top utterances
    for k,v in sorted(people.items(), key=lambda i:sum(i[1]), reverse=True):
        fsave_name = '_'.join(k.lower().split(' '))
        handle = open_for_write("stats/" + settings['prefix'] + "/top_utterances/" + fsave_name)
        for k2,v2 in sorted(other_utt_dict[k].items(), key=operator.itemgetter(1), reverse=True):
            if v2 == 1:
                break
            handle.write(k2 + '\t' + str(v2) + '\n')
        handle.close()

    print("\nOutput conversation lengths: ")
    table = PrettyTable(("Name\t" + "\t".join(MSG_TYPE.keys()) + "\tO(M)\tI(M)\tO(W)\tI(W)\tOut Ratio\tIn Ratio").split("\t"))
    table.align["Name"] = "l"
    handle = open_for_write("stats/" + settings['prefix'] + "/total_messages_per_person.csv")
    handle.write("Name\t" + "\t".join(MSG_TYPE.keys()) + "\tO(M)\tI(M)\tO(W)\tI(W)\tOut Ratio\tIn Ratio\n")
    for k,v in sorted(people.items(), key=lambda i:sum(i[1]), reverse=True):
        if k in settings['my_name']:
            continue
        table_row = str(k) + "\t" + "\t".join([str(v2) for v2 in v]) + "\t" + str(people_parts[k]["outgoing"]) + "\t" + str(people_parts[k]["incoming"]) + "\t" + str(people_parts[k]["out_words"]) + "\t" + str(people_parts[k]["in_words"]) + "\t" + "{:2.2f}".format(people_parts[k]["out_words"]*1.0/people_parts[k]["outgoing"] if people_parts[k]["outgoing"] > 0 else 0) + "\t" + "{:2.2f}".format(people_parts[k]["in_words"]*1.0/people_parts[k]["incoming"] if people_parts[k]["incoming"] > 0 else 0)
        handle.write(table_row + "\n")
        table.add_row(table_row.split("\t"))
    handle.close()
    print(table.get_string(sort_key=operator.itemgetter(0), sortby="Name"))

    print("\nCategory output lengths: ")
    table2 = PrettyTable(["Tag", "Value", "I(W)", "O(W)", "T(W)", "I(M)", "O(M)", "T(M)"])
    table2.align["Tag"] = "l"
    table2.align["Value"] = "l"
    handle = open_for_write("stats/" + settings['prefix'] + "/total_messages_per_category.csv")
    handle.write("Tag\tValue\tI(W)\tO(W)\tT(W)\tI(M)\tO(M)\tT(M)\n")
    for tag in tag_set:
        for tval in tag_set[tag]:
            in_msg_num = sum([people_group["incoming"][tag][tval]["chrono"][mdate]["messages"] for mdate in people_group["incoming"][tag][tval]["chrono"]])
            out_msg_num = sum([people_group["outgoing"][tag][tval]["chrono"][mdate]["messages"] for mdate in people_group["outgoing"][tag][tval]["chrono"]])
            in_word_num = sum([people_group["incoming"][tag][tval]["chrono"][mdate]["words"] for mdate in people_group["incoming"][tag][tval]["chrono"]])
            out_word_num = sum([people_group["outgoing"][tag][tval]["chrono"][mdate]["words"] for mdate in people_group["outgoing"][tag][tval]["chrono"]])
            handle.write(tag + "\t" + tval + "\t" + str(in_word_num) + "\t" + str(out_word_num) + "\t" + str(in_word_num + out_word_num) + "\t" + str(in_msg_num) + "\t" + str(out_msg_num) + "\t" + str(in_msg_num + out_msg_num) + "\n")
            table2.add_row([tag, tval, in_word_num, out_word_num, in_word_num + out_word_num, in_msg_num, out_msg_num, in_msg_num + out_msg_num])
    handle.write("\n\n")
    for tag in tag_set:
        for tval in tag_set[tag]:
            handle.write(tag + "\t" + tval + "\t")
            tppl = "\t\t"
            for k,v in sorted(people.items(), key=lambda i:sum(i[1]), reverse=True):
                if k in settings['my_name']:
                    continue
                if valid_people[k][tag] == tval:
                    tppl += "\t" + str(k)
                    handle.write("\t" + str(people_parts[k]["out_words"] + people_parts[k]["in_words"]))
            handle.write("\n")
            #handle.write(tppl + "\n")
    handle.close()
    print(table2.get_string(sort_key=operator.itemgetter(0, 1), sortby="Tag"))

    my_outgoing_msgs = sum([people_parts[k]["outgoing"] for k in people_parts.keys()])
    my_outgoing_words = sum([people_parts[k]["out_words"] for k in people_parts.keys()])
    print("My ratio: " + str(my_outgoing_words*1.0/my_outgoing_msgs))

    #for ind in range(len(liwc_keys)):
    #    print("For LIWC key " + liwc_keys[ind] + ":")
    #    for person in people_parts.keys():
    #        print(person + " (" + str(len(people_parts[person]["liwc_words"][ind])) + "): " + str(set(people_parts[person]["liwc_words"][ind])))
    #    print("\n")

    print("\nTop words from me: " + str(get_top_words(token_me)))
    print("\nTop words from others: " + str(get_top_words(token_other)))
    write_month_chronology(people, people_parts)
    write_liwc_chronology(people.items(), people_parts, people_group)
    write_time_stats(people, people_parts, people_group)

def make_person():
    temp_parts = {}
    temp_parts["incoming"] = 0
    temp_parts["outgoing"] = 0
    temp_parts["in_words"] = 0
    temp_parts["out_words"] = 0
    temp_parts["chrono"] = {y + ":" + m : {"words": 0, "messages": 0} for y in years for m in months}
    temp_parts["response_times"] = {key: [] for key in temp_parts["chrono"].keys()}
    temp_parts["liwc_counts"] = {key: np.array([0]*len(liwc_keys)) for key in temp_parts["chrono"].keys()}
    temp_parts["liwc_words"] = [[] for i in range(len(liwc_keys))]
    return temp_parts

# @deprecated
def make_heat_maps(people_group):
    for inout in ["incoming", "outgoing"]:
        pgr_norm = {tag: {tval: [0]*len(liwc_keys) for tval in tag_set[tag]} for tag in tag_set}
        for tag in tag_set:
            for tval in tag_set[tag]:
                for i in range(0, len(liwc_keys)):
                    l_cat = liwc_keys[i]
                    f_value = sum([people_group[inout][tag][tval]["liwc_counts"][_d][i] for _d in [y + ":" + m for y in years for m in months]]) * 1.0
                    f_denom = sum([people_group[inout][tag][tval]["chrono"][_d]["words"] for _d in [y + ":" + m for y in years for m in months]])
                    f_value = f_value * 1.0 / f_denom if f_denom > 0 else 0
                    pgr_norm[tag][tval][i] = f_value
        xticks = []
        yticks = []
        ndy = True

        #print("PGR: " + str(pgr_norm["family"]["yes"]))
        for tag in tag_set:
            for tval in tag_set[tag]:
                xticks.append(tag + ": " + tval)
                for i in range(0, len(liwc_keys)):
                    if ndy:
                        yticks.append(liwc_keys[i])
                    lag_cat = [pgr_norm[tag][tval][i] for tag in tag_set for tval in tag_set[tag]]
                    pgr_norm[tag][tval][i] = (pgr_norm[tag][tval][i] - min(lag_cat)) / max(lag_cat)
                ndy = False

        npheat = np.zeros((len(xticks), len(yticks)))
        xind, yind = [0]*2
        for tag in tag_set:
            for tval in tag_set[tag]:
                yind = 0
                for i in range(0, len(liwc_keys)):
                    npheat[xind][yind] = pgr_norm[tag][tval][i]
                    yind += 1
                xind += 1

        plt.title(inout)
        plt.imshow(npheat, cmap='hot', interpolation='nearest')
        plt.xticks(range(len(yticks)), yticks)
        plt.xticks(rotation=90)
        plt.yticks(range(len(xticks)), xticks)
        plt.show()

def write_month_chronology(pf, ppts):
    handle = open_for_write("stats/" + settings['prefix'] + "/msgs_over_time_by_person.csv")
    # write header
    handle.write("Name")
    for _d in [y + ":" + m for y in years for m in months]:
        handle.write("\t" + _d)
    handle.write("\n")
    # write user data by sum of messages per user descending
    for k,v in sorted(pf.items(), key=lambda i:sum(i[1]), reverse=True):
        handle.write(k)
        # write in date order
        handle.write("\t" + "\t".join([str(ppts[k]["chrono"][k2]["messages"]) for k2 in sorted(ppts[k]["chrono"].keys())]) + "\n")

def write_time_stats(people, ppts, group):
    h1 = open_for_write("stats/" + settings['prefix'] + "/freq_by_stat/messages_per_month.csv")
    h2 = open_for_write("stats/" + settings['prefix'] + "/freq_by_stat/words_per_month.csv")
    h3 = open_for_write("stats/" + settings['prefix'] + "/freq_by_stat/wpm_per_month.csv")
    # Write header
    h1.write("Name\t" + "\t".join([y + ":" + m for y in years for m in months]) + "\n")
    h2.write("Name\t" + "\t".join([y + ":" + m for y in years for m in months]) + "\n")
    h3.write("Name\t" + "\t".join([y + ":" + m for y in years for m in months]) + "\n")
    # Write user data by sum of messages per user descending
    for k,v in sorted(people.items(), key=lambda i:sum(i[1]), reverse=True):
        # Write in date order
        h1.write(str(k) + "\t" + "\t".join([str(ppts[k]["chrono"][k2]["messages"]) for k2 in sorted(ppts[k]["chrono"].keys())]) + "\n")
        h2.write(str(k) + "\t" + "\t".join([str(ppts[k]["chrono"][k2]["words"]) for k2 in sorted(ppts[k]["chrono"].keys())]) + "\n")
        h3.write(str(k) + "\t" + "\t".join([str(ppts[k]["chrono"][k2]["words"]*1.0/ppts[k]["chrono"][k2]["messages"] if ppts[k]["chrono"][k2]["messages"] > 0 else 0) for k2 in sorted(ppts[k]["chrono"].keys())]) + "\n")
    h1.close()
    h2.close()
    h3.close()

    h1 = open_for_write("stats/" + settings['prefix'] + "/freq_by_stat/group_messages_per_month.csv")
    h2 = open_for_write("stats/" + settings['prefix'] + "/freq_by_stat/group_words_per_month.csv")
    h3 = open_for_write("stats/" + settings['prefix'] + "/freq_by_stat/group_wpm_per_month.csv")
    # Write header
    h1.write("Tag\tValue\t" + "\t".join([y + ":" + m for y in years for m in months]) + "\n")
    h2.write("Tag\tValue\t" + "\t".join([y + ":" + m for y in years for m in months]) + "\n")
    h3.write("Tag\tValue\t" + "\t".join([y + ":" + m for y in years for m in months]) + "\n")
    for tag in tag_set:
        for tval in tag_set[tag]:
            h1.write(str(tag) + "\t" + str(tval) + "\t" + "\t".join([str(group["incoming"][tag][tval]["chrono"][mdate]["messages"] + group["outgoing"][tag][tval]["chrono"][mdate]["messages"]) for mdate in sorted(group["incoming"][tag][tval]["chrono"])]) + "\n")
            h2.write(str(tag) + "\t" + str(tval) + "\t" + "\t".join([str(group["incoming"][tag][tval]["chrono"][mdate]["words"] + group["outgoing"][tag][tval]["chrono"][mdate]["words"]) for mdate in sorted(group["incoming"][tag][tval]["chrono"])]) + "\n")
            h3.write(str(tag) + "\t" + str(tval) + "\t" + "\t".join([str(((group["incoming"][tag][tval]["chrono"][mdate]["words"] + group["outgoing"][tag][tval]["chrono"][mdate]["words"])*1.0 / (group["incoming"][tag][tval]["chrono"][mdate]["messages"] + group["outgoing"][tag][tval]["chrono"][mdate]["messages"])) if group["incoming"][tag][tval]["chrono"][mdate]["messages"] + group["outgoing"][tag][tval]["chrono"][mdate]["messages"] > 0 else 0) for mdate in sorted(group["incoming"][tag][tval]["chrono"])]) + "\n")
    h1.close()
    h2.close()
    h3.close()

    # Write response_time / time
    h1 = open_for_write("stats/" + settings['prefix'] + "/freq_by_stat/group_resptime_per_month.csv")
    # Write header
    h1.write("Tag\tValue\t" + "\t".join([y + ":" + m for y in years for m in months]) + "\n")
    for tag in tag_set:
        for tval in tag_set[tag]:
            h1.write(str(tag) + "\t" + str(tval) + "\t" + "\t".join([str(np.average(group["outgoing"][tag][tval]["response_times"][mdate])) for mdate in sorted(group["outgoing"][tag][tval]["response_times"])]) + "\n")
    h1.close()

def write_liwc_chronology(people, ppts, group):
    # Write CSV by category
    for i in range(0, len(liwc_keys)):
        l_cat = liwc_keys[i]
        handle = open_for_write("stats/" + settings['prefix'] + "/liwc_by_category/" + l_cat.lower() + ".csv")
        # Write header
        handle.write("Name")
        for _d in [y + ":" + m for y in years for m in months]:
            handle.write("\t" + _d)
        handle.write("\n")
        # Write user data
        for name in people:
            t_name = name[0]
            handle.write(t_name)
            for _d in [y + ":" + m for y in years for m in months]:
                f_value = ppts[t_name]["liwc_counts"][_d][i] * 1.0
                f_value = f_value * 1.0 / ppts[t_name]["chrono"][_d]["words"] if ppts[t_name]["chrono"][_d]["words"] > 0 else 0
                handle.write("\t" + str(f_value))
            handle.write("\n")
        handle.close()

    # Write CSV by person
    for name in people:
        t_name = name[0]
        handle = open_for_write("stats/" + settings['prefix'] + "/liwc_by_person/" + "_".join(t_name.strip().split()) + ".csv")
        # Write header
        handle.write("Category")
        for _d in [y + ":" + m for y in years for m in months]:
            handle.write("\t" + _d)
        handle.write("\n")
        # Write user data
        for i in range(0, len(liwc_keys)):
            l_cat = liwc_keys[i]
            handle.write(l_cat)
            for _d in [y + ":" + m for y in years for m in months]:
                f_value = ppts[t_name]["liwc_counts"][_d][i] * 1.0
                f_value = f_value * 1.0 / ppts[t_name]["chrono"][_d]["words"] if ppts[t_name]["chrono"][_d]["words"] > 0 else 0
                handle.write("\t" + str(f_value))
            handle.write("\n")
        handle.close()

    # Write CSV that is people by counts
    handle = open_for_write("stats/" + settings['prefix'] + "/liwc_person.csv")
    handle.write("Name\tWords\t" + "\t".join([liwc_keys[i] for i in range(0, len(liwc_keys))]) + "\n")
    for name in people:
        t_name = name[0]
        tp_words = sum([ppts[t_name]["chrono"][_d]["words"] for _d in [y + ":" + m for y in years for m in months]])
        handle.write(t_name + '\t' + str(tp_words))
        for i in range(0, len(liwc_keys)):
            l_cat = liwc_keys[i]
            # Write user data
            f_value = sum([ppts[t_name]["liwc_counts"][_d][i] for _d in [y + ":" + m for y in years for m in months]])
            handle.write("\t" + str(f_value))
        handle.write("\n")
    handle.close()

    # Write CSV that is people by counts
    handle = open_for_write("stats/" + settings['prefix'] + "/liwc_person_normalized.csv")
    handle.write("Name\t" + "\t".join([liwc_keys[i] for i in range(0, len(liwc_keys))]) + "\n")
    for name in people:
        t_name = name[0]
        tp_words = sum([ppts[t_name]["chrono"][_d]["words"] for _d in [y + ":" + m for y in years for m in months]])
        handle.write(t_name)# + '\t' + str(tp_words))
        for i in range(0, len(liwc_keys)):
            l_cat = liwc_keys[i]
            # Write user data
            f_value = sum([ppts[t_name]["liwc_counts"][_d][i] for _d in [y + ":" + m for y in years for m in months]])
            handle.write("\t" + str(f_value*1.0/tp_words))
        handle.write("\n")
    handle.close()

    # Write CSV by group
    for i in range(0, len(liwc_keys)):
        l_cat = liwc_keys[i]
        handle = open_for_write("stats/" + settings['prefix'] + "/liwc_by_group/" + l_cat.lower() + ".csv")
        # Write header
        handle.write("Tag\tValue\tIn/Out")
        for _d in [y + ":" + m for y in years for m in months]:
            handle.write("\t" + _d)
        handle.write("\n")

        for tag in tag_set:
            for tval in tag_set[tag]:
                for inout in ["incoming", "outgoing"]:
                    handle.write(str(tag) + "\t" + str(tval) + "\t" + str(inout))
                    for _d in [y + ":" + m for y in years for m in months]:
                        f_value = group[inout][tag][tval]["liwc_counts"][_d][i] * 1.0
                        f_value = f_value * 1.0 / group[inout][tag][tval]["chrono"][_d]["words"] if group[inout][tag][tval]["chrono"][_d]["words"] > 0 else 0
                        handle.write("\t" + str(f_value))
                    handle.write("\n")
        handle.close()

    # Write CSV that is groups by counts
    handle = open_for_write("stats/" + settings['prefix'] + "/liwc_grouped.csv")
    for count_name in ["Normalized Counts", "Raw Counts"]:
        handle.write("Tag\tValue\tIn/Out\t" + "\t".join([liwc_keys[i] for i in range(0, len(liwc_keys))]) + "\n")
        for tag in tag_set:
            for tval in tag_set[tag]:
                for inout in ["incoming", "outgoing"]:
                    handle.write(str(tag) + "\t" + str(tval) + "\t" + str(inout))
                    for i in range(0, len(liwc_keys)):
                        l_cat = liwc_keys[i]
                        f_value = sum([group[inout][tag][tval]["liwc_counts"][_d][i] for _d in [y + ":" + m for y in years for m in months]]) * 1.0
                        if count_name == "Normalized Counts":
                            f_denom = sum([group[inout][tag][tval]["chrono"][_d]["words"] for _d in [y + ":" + m for y in years for m in months]])
                            f_value = f_value * 1.0 / f_denom if f_denom > 0 else 0
                        handle.write("\t" + str(f_value))
                    handle.write("\n")
        handle.write("\n")
    handle.close()

def get_top_words(token_set, TOP_WORDS=50):
    top_tokens = []
    for k,v in sorted(token_set.items(), key=operator.itemgetter(1), reverse=True)[:TOP_WORDS]:
        top_tokens.append(k)
    return top_tokens

if __name__ == "__main__":
    main()
