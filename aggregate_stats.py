

import operator, msgpack, nltk, math, sys, os
from prettytable import PrettyTable
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm

import numpy as np
import dateutil.parser

from utils import START_TIME, settings, liwc_keys, open_for_write, month_to_season, season_name, month_name
#from normalizer import expand_text, norms
from user_annotator import read_people_file, tag_set

seasons = range(1, 13)

def main():
    print('Do not run this file.')

def get_agg_stats(years, month_bin):
    months = ["0" + str(a) if len(str(a)) < 2 else str(a) for a in range(1, 13, month_bin)]

    valid_people = read_people_file("tagged_people")
    all_msgs = {y + ":" + m : {"messages": 0, "people": set()} for y in years for m in months}
    hour_of_day = {hr: 0 for hr in range(0, 24)}
    day_of_week = {day: 0 for day in range(0, 7)}
    hod_group = {tag: {tval: {hr: 0 for hr in range(0, 24)} for tval in tag_set[tag]} for tag in tag_set}
    dow_group = {tag: {tval: {day: 0 for day in range(0, 7)} for tval in tag_set[tag]} for tag in tag_set}
    season_dist = {sea: 0 for sea in seasons}

    print("Reading conversation files...")
    for filename in os.listdir(settings['DATA_DIR']):
        print("File: " + filename)
        convo = None
        with open(settings['DATA_DIR'] + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()

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

            mdate = dateutil.parser.parse(message[b"date"])
            season_dist[mdate.month] += 1
            hour_of_day[mdate.hour] += 1
            day_of_week[mdate.weekday()] += 1
            for tag in tag_set:
                if cname in valid_people:
                    hod_group[tag][valid_people[cname][tag]][mdate.hour] += 1
                    dow_group[tag][valid_people[cname][tag]][mdate.weekday()] += 1
            mmstr = str(((mdate.month-1) // month_bin) * month_bin + 1)
            mdate = str(mdate.year) + ":" + ("0" + mmstr if len(mmstr) < 2 else mmstr)

            all_msgs[mdate]["messages"] += 1
            all_msgs[mdate]["people"].add(cname)

            # add up tokens and number of incoming and outgoing messages
            #tokens = [_t for _t in nltk.word_tokenize(msg_text)]

            if message[b"user"].decode() in settings['my_name']:# if the person is me
                pass
            else:# count tokens from other person
                pass

    # write month and season dist
    handle = open_for_write('stats/' + settings['prefix'] + '/month_season_dists.txt')
    season_sum = {sea: 0 for sea in season_name.values()}
    for i in season_dist:
        handle.write(month_name[i] + '\t' + str(season_dist[i]) + '\n')
        season_sum[season_name[month_to_season[i]]] += season_dist[i]
    for k,v in season_sum.items():
        handle.write(k + '\t' + str(v) + '\n')

    # write other files
    write_hod(hour_of_day, hod_group)
    write_dow(day_of_week, dow_group)
    write_month_chronology(all_msgs, years, months)

def write_hod(hod, hod_group):
    handle = open_for_write("stats/" + settings['prefix'] + "/hour_of_day_stats.csv")
    handle.write("Type")
    for hr in range(0, 24):
        handle.write("\t" + str(hr))
    handle.write("\nAll")
    for hr in range(0, 24):
        handle.write("\t" + str(hod[hr]))
    for tag in tag_set:
        handle.write("\n" + tag)
        for tval in tag_set[tag]:
            handle.write("\n" + tval)
            for hr in range(0, 24):
                hr_num = hod_group[tag][tval][hr]*1.0
                hr_denom = sum(hod_group[tag][tval].values())
                handle.write("\t" + str(hr_num / hr_denom if hr_denom > 0 else 0))

def write_dow(dow, dow_group):
    handle = open_for_write("stats/" + settings['prefix'] + "/day_of_week_stats.csv")
    handle.write("Type")
    for hr in range(0, 7):
        handle.write("\t" + str(hr))
    handle.write("\nAll")
    for hr in range(0, 7):
        handle.write("\t" + str(dow[hr]))
    for tag in tag_set:
        handle.write("\n" + tag)
        for tval in tag_set[tag]:
            handle.write("\n" + tval)
            for hr in range(0, 7):
                hr_num = dow_group[tag][tval][hr]*1.0
                hr_denom = sum(dow_group[tag][tval].values())
                handle.write("\t" + str(hr_num / hr_denom if hr_denom > 0 else 0))

def write_month_chronology(all_msgs, years, months):
    handle = open_for_write("stats/" + settings['prefix'] + "/msgs_aggregate.csv")
    # write header
    handle.write("Type")
    for _d in [y + ":" + m for y in years for m in months]:
        handle.write("\t" + _d)
    handle.write("\n")
    # write in date order
    handle.write("All Messages")
    handle.write("\t" + "\t".join([str(all_msgs[k]["messages"]) for k in sorted(all_msgs.keys())]) + "\n")
    handle.write("Number of People")
    handle.write("\t" + "\t".join([str(len(all_msgs[k]["people"])) for k in sorted(all_msgs.keys())]) + "\n")

if __name__ == "__main__":
    main()
