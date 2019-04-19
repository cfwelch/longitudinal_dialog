

import msgpack, nltk, math, os
import dateutil.parser

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import DEFAULT_TIMEZONE, settings

SPLIT_INTO = 5

# To split into 5 bins we will need 39315.6 points per bin.
# The bin cutoffs are: [-1, 7, 21, 59, 256, 266937297]
# 	Bin 1: 39874
# 	Bin 2: 40384
# 	Bin 3: 38101
# 	Bin 4: 38925
# 	Bin 5: 39294

def main():
    # Read all the data
    list_dir_set = os.listdir(settings['DATA_MERGE_DIR'])
    response_times = []
    count = 0
    zeros = 0
    for filename in list_dir_set:
        count += 1
        print('File (' + str(count) + '/' + str(len(list_dir_set)) + '): ' + filename)
        convo = None
        with open(settings['DATA_MERGE_DIR'] + '/' + filename, 'rb') as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b'with'].decode()

        prev = None
        prev_date = None
        for message in tqdm(convo[b'messages']):
            if b'text' not in message:
                continue
            cur_speaker = message[b'user'].decode()
            mdate = dateutil.parser.parse(message[b'date'])
            if cur_speaker in settings['my_name'] and prev != None and prev not in settings['my_name']:
                #msg_text = message[b'text']
                if mdate.tzinfo == None:
                    mdate = DEFAULT_TIMEZONE.localize(mdate)
                td = (mdate - prev_date).seconds + (mdate - prev_date).days*24*60*60 if prev_date != None else 0
                if td > 0:
                    response_times.append(td)
                else:
                    zeros += 1
            prev = cur_speaker
            prev_date = mdate

    print('Number of points where turn changes to you: ' + str(len(response_times)))
    #print('Number of times where td=0: ' + str(zeros))
    pts_per_bin = len(response_times) * 1.0 / SPLIT_INTO
    print('To split into ' + str(SPLIT_INTO) + ' bins we will need ' + str(pts_per_bin) + ' points per bin.')

    rps = np.array(response_times)
    bin_counter = 0
    cutoffs = [-1]
    for secs in range(0, max(response_times)):
        if np.sum(rps <= secs) >= (bin_counter + 1) * pts_per_bin:
            cutoffs.append(secs)
            bin_counter += 1
            if len(cutoffs) == SPLIT_INTO:
                break
    cutoffs.append(max(response_times))

    print('The bin cutoffs are: ' + str(cutoffs))
    for i in range(SPLIT_INTO):
        print('\tBin ' + str(i+1) + ': ' + str(np.sum((rps > cutoffs[i]) & (rps <= cutoffs[i+1]))))

    #response_times = [math.log(i) for i in response_times]
    #plt.hist(response_times, bins=100)
    #plt.show()

if __name__ == '__main__':
    main()
