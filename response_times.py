

import msgpack, operator, nltk, math, json, re, os

from collections import Counter
from tqdm import tqdm
from pprint import pprint
from datetime import datetime

import dateutil.parser
#import matplotlib.pyplot as plt
import numpy as np

from utils import settings, open_for_write, tag_set, read_people_file

def main():
    print('Do not run this file. It is called from get_stats.py with the -gr flag.')

def get_rt_stats():
    valid_people = read_people_file('tagged_people')
    people_group = {inout: {tag: {tval: [] for tval in tag_set[tag]} for tag in tag_set} for inout in ['incoming', 'outgoing']}
    # incoming means to me, outgoing means from me
    rt_me = []
    np_rt_me = []
    rt_ot = []

    print('Reading conversation files...')
    for filename in os.listdir(settings['DATA_MERGE_DIR']):
        print('File: ' + filename)
        convo = None
        with open(settings['DATA_MERGE_DIR'] + '/' + filename, 'rb') as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b'with'].decode()
        if cname not in valid_people:
            print(cname + ' is not in the list of tagged people...')
            continue

        for message in tqdm(convo[b'messages']):
            if b'text' not in message:
                continue
            msg_text = message[b'text']
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()

            mdate = dateutil.parser.parse(message[b'date'])

            #if mdate.year not in [2016, 2017, 2018]:
            #if mdate.year == 2016:
            if message[b'turn_change']:
                if message[b'user'].decode() in settings['my_name']:
                    rt_me.append(math.log(message[b'response_time']) if message[b'response_time'] > 0 else 0)
                    #rt_me.append(message[b'response_time'])
                    np_rt_me.append(message[b'response_time'])
                    for tag in tag_set:
                        people_group['outgoing'][tag][valid_people[cname][tag]].append(message[b'response_time'])
                else:
                    rt_ot.append(message[b'response_time'])
                    for tag in tag_set:
                        people_group['incoming'][tag][valid_people[cname][tag]].append(message[b'response_time'])

    #for bins in [50, 100]:#[5, 10, 20, 30]
    #    hist_pts = plt.hist(rt_me, bins, log=True)#, normed=1
    #    #plt.yscale('log', nonposy='clip')
    #    print(len(hist_pts))
    #    print(hist_pts)
    #    print('\n')
    #    plt.show()

    #plt.hist(rt_ot, 100)
    #plt.yscale('log', nonposy='clip')
    #plt.show()

    nprtme = np.array(np_rt_me)
    #for bins in [60, 300, 1800, 3600, 86400, 604800, 2592000, 31557600, 157788000]:
    prev_bin = 0
    handle = open_for_write('stats/' + settings['prefix'] + '/response_time_bins.txt')
    for bins in [60, 300, 1800, 86400, 2419200, 31557600]:
        new_bin = sum(nprtme<bins)
        print(str(bins) + ': ' + str(new_bin))
        handle.write(str(bins) + '\t' + str(new_bin - prev_bin) + '\n')
        prev_bin = new_bin
    handle.close()

    rt_stats(rt_me, 'Me')
    rt_stats(rt_ot, 'Other')
    print('\n\n')
    for inout in ['incoming', 'outgoing']:
        for tag in tag_set:
            for tval in tag_set[tag]:
                avg_rt = np.average(people_group[inout][tag][tval])
                median_rt = np.median(people_group[inout][tag][tval])
                print(inout + '\t' + tag + '\t' + tval + '\t' + str(median_rt))

def rt_stats(rt, ostr):
    rt = np.array(rt)
    avg_rt = np.average(rt)
    print('\n' + '='*15 + ostr + '='*15)
    print('Average: ' + str(avg_rt))
    print('Max: ' + str(np.max(rt)))
    print('Min: ' + str(np.min(rt)))
    print('Std: ' + str(np.std(rt)))
    rmed = np.median(rt)
    print('Median: ' + str(rmed))
    print('Points Above Median: ' + str(len(np.where(rt>rmed)[0])))
    print('Points Below Median: ' + str(len(np.where(rt<=rmed)[0])))

    sumsofar = 0
    for i in range(24):
        thispart = len(np.where(rt<60*60*(i+1))[0])
        print('Points below ' + str(i+1) + ' hours: ' + str(thispart - sumsofar))
        sumsofar = thispart

    return avg_rt

if __name__ == '__main__':
    main()
