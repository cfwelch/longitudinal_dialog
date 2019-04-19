

import datetime, msgpack, random, shutil, nltk, copy, math, sys, os
import dateutil.parser
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

from utils import SPTOK_ME, SPTOK_NU, SPTOK_OTHER, SPTOK_UNK, liwc_keys, month_to_season, open_for_write, DEFAULT_TIMEZONE, START_TIME, settings
from make_splits import convert_prt

# This file takes a split file and creates an experiment folder
# This folder can be transferred and run on another machine and has no vocab file
def main():
    parser = ArgumentParser()
    parser.add_argument('-py', '--python-only', dest='pyonly', help='Only copy the python source', default=False, action='store_true')
    parser.add_argument('-g', '--gvp', dest='gvp', help='There is an additional general training set', default=False, action='store_true')
    parser.add_argument('-t', '--task', dest='task', help='Problem formulation is response time (rt), user attribute (ua), common utterance classification (cu), all utterance generation (ag which is for seq2seq), or next word prediction (nw)', default='', type=str)
    parser.add_argument('-u', '--user', dest='user', help='Decide which user should be used for prediction. Input their name. Default is settings[\'my_name\'] from utils.py', default=settings['my_name'], type=str)
    opt = parser.parse_args()

    embed_sets = ['glove_']

    if opt.task == '':
        print('Select response time (rt), user attributes (ua), common utterance classification (cu).')
        sys.exit(0)

    t_my_name = opt.user[0] if type(opt.user) == list else opt.user
    fsave_name = '_'.join(t_my_name.lower().split(' '))
    now = datetime.datetime.now()
    fname = 'experiment/exp_' + str(now.year) + '_' + '{:02d}'.format(now.month) + '_' + '{:02d}'.format(now.day) + '_T_' + '{:02d}'.format(now.hour) + '_' + '{:02d}'.format(now.minute) + '_' + '{:02d}'.format(now.second)
    os.makedirs(fname)

    if not opt.pyonly:
        means, stds = rewrite_context('train', fname, opt.task, fsave_name, embed_sets=embed_sets)
        if opt.gvp:
            rewrite_context('gentr', fname, opt.task, fsave_name, norm=[means, stds], embed_sets=embed_sets)
        rewrite_context('dev', fname, opt.task, fsave_name, norm=[means, stds], embed_sets=embed_sets)
        rewrite_context('test', fname, opt.task, fsave_name, norm=[means, stds], embed_sets=embed_sets)

    for pyname in os.listdir('model'):
        if pyname.endswith('.py'):
            shutil.copy('model/' + pyname, fname + '/' + pyname)

    shutil.copy('utils.py', fname + '/utils.py')
    shutil.copy('ctx_sizes.py', fname + '/ctx_sizes.py')
    os.makedirs(fname + '/config')
    shutil.copy('config/' + settings['prefix'] + '.cfg', fname + '/config/' + settings['prefix'] + '.cfg')
    if not opt.pyonly:
        for es in embed_sets:
            shutil.copy('splits/' + settings['prefix'] + '/' + fsave_name + '/' + es + 'embeds.th', fname + '/' + es + 'embeds.th')
            shutil.copy('splits/' + settings['prefix'] + '/' + fsave_name + '/' + es + 'vocabulary', fname + '/' + es + 'vocabulary')

    shutil.make_archive(fname, 'tar', fname)
    shutil.rmtree(fname)

def rewrite_context(section, fname, task, fsave_name, norm=None, embed_sets=None):
    # Take no more than MAX_PER_NW instances of a word when creating splits
    MAX_PER_NW = 9999
    data_set = None
    unk_count = 0
    tu_tok = 0

    vocabs = {i: {} for i in embed_sets}
    for es in embed_sets:
        with open('splits/' + settings['prefix'] + '/' + fsave_name + '/' + es + 'vocabulary', 'rb') as f:
            t_vocab = msgpack.unpackb(f.read())
            for key in t_vocab:
                vocabs[es][key.decode()] = t_vocab[key]
        print('Vocab Length: ' + str(len(vocabs[es])))

    if norm == None:
        user_vecs, liwc_vecs, time_vecs, freq_vecs, turn_vecs, stop_vecs, lsm_vecs, surf_vecs = [], [], [], [], [], [], [], []
    else:
        user_mean, liwc_mean, time_mean, freq_mean, turn_mean, stop_mean, lsm_mean, surf_mean = norm[0]
        user_std, liwc_std, time_std, freq_std, turn_std, stop_std, lsm_std, surf_std = norm[1]

    # read dataset
    print('-'*20 + '\nLooking at section ' + section + '...')
    with open('splits/' + settings['prefix'] + '/' + fsave_name + '/' + section + '_contexts', 'rb') as f:
        data_set = msgpack.unpackb(f.read())

        ######### Parts #########
        #  0: Full date of message sent (b'date')
        #  1: <ME> or <OTHER> (b'user')
        #  2: Message text (b'text')
        #  3: LIWC counts (b'liwc_counts' -- don't use b'liwc_words')
        #  4: User vector representation from tag set -- comes from tagged_people file
        #  5: Response time in seconds (b'response_time')
        #  6: Frequencies of communication in order: (b'all_freq', b'month_freq', b'week_freq', b'day_freq')
        #  7: Turn changes (b'turn_change')
        #  8: Stop word count (b'stopword_count')
        #  9: Linguistic Style Matching (b'lsm')
        # 10: Unique index representing other speaker
        # 11: Mention graph counts (b'mentions')
        #########################
        skipped_nw = 0
        dialog_lengths = []
        nw_counts = defaultdict(lambda: 0)
        chosen_points = []
        for point in tqdm(range(0, len(data_set))):
            dialog = {es: [vocabs[es][SPTOK_NU]] for es in embed_sets}
            liwc_words = np.array([0]*len(liwc_keys))
            liwc_me = np.array([0]*len(liwc_keys))
            liwc_other = np.array([0]*len(liwc_keys))
            first_date = None
            response_time = None
            ctx_words, ctx_me, ctx_other = [0, 0, 0]
            sym_all, sym_me, sym_other = [0, 0, 0]
            mention_sum = data_set[point][-1][11] #np.array([0]*len())

            if task == 'rt':
                last_date = dateutil.parser.parse(data_set[point][-2][0]) if data_set[point][-2] != None else None
            else: # cu and ua
                last_date = dateutil.parser.parse(data_set[point][-1][0])
            if last_date != None and last_date.tzinfo == None:
                last_date = DEFAULT_TIMEZONE.localize(last_date)

            if task == 'ua':
                feature_window = data_set[point]
            else:
                feature_window = data_set[point][:-1]

            message_diffs = []
            previous_date = DEFAULT_TIMEZONE.localize(START_TIME)
            freqs = [None]*4
            freqs_end = [None]*4
            stop_me, stop_other = [0, 0]
            utt_me, utt_other = [0, 0]
            for context_vector in feature_window:
                if context_vector == None:
                    message_diffs.append(24*60*60*365)
                    continue
                cur_date = dateutil.parser.parse(context_vector[0])
                if cur_date.tzinfo == None:
                    cur_date = DEFAULT_TIMEZONE.localize(cur_date)
                if first_date == None:
                    first_date = cur_date
                if freqs[0] == None:
                    freqs = context_vector[6]

                #if previous_date != None:
                td = cur_date - previous_date
                td = td.days*24*60*60 + td.seconds
                message_diffs.append(td)

                msg_text = context_vector[2].decode()
                #utter = msg_text.split(' ')
                utter = [_t.lower() for _t in nltk.word_tokenize(msg_text)]
                utter.insert(0, context_vector[1].decode())
                if context_vector[1].decode() == SPTOK_ME:
                    utt_me += 1
                    stop_me += context_vector[8]
                    liwc_me += context_vector[3]
                    ctx_me += len(utter)
                else:
                    utt_other += 1
                    stop_other += context_vector[8]
                    liwc_other += context_vector[3]
                    ctx_other += len(utter)

                freqs_end = context_vector[6]
                response_time = cur_date
                liwc_words += context_vector[3]
                #mention_sum += context_vector[11] # You can't do this because you don't know how to sum windows...
                ctx_words += len(utter)
                for token in range(0, len(utter)):
                    tu_tok += 1
                    sym_all += len(utter[token])
                    if context_vector[1].decode() == SPTOK_ME:
                        sym_me += len(utter[token])
                    else:
                        sym_other += len(utter[token])
                    try:
                        for es in embed_sets:
                            dialog[es].append(vocabs[es][utter[token]])
                    except KeyError:
                        for es in embed_sets:
                            dialog[es].append(vocabs[es][SPTOK_UNK])
                        unk_count += 1
                previous_date = cur_date

            if ctx_words > 0:
                liwc_words = liwc_words / ctx_words
            if ctx_me > 0:
                liwc_me = liwc_me / ctx_me
            if ctx_other > 0:
                liwc_other = liwc_other / ctx_other
            for es in embed_sets:
                assert max(dialog[es]) <= len(vocabs[es])
            #print('last date: ' + str(last_date))
            #print('first date: ' + str(first_date))

            last_utt = data_set[point][-1][2].decode()
            last_utt = [_t.lower() for _t in nltk.word_tokenize(last_utt)]

            split_point = None
            truncated_utt = {es: [vocabs[es][SPTOK_NU]] for es in embed_sets}
            if task == 'nw':
                if len(last_utt) == 0:
                    skipped_nw += 1
                    continue
                split_point = random.choice(range(len(last_utt)))
                orig_split = split_point
                split_point = (split_point + 1) % len(last_utt)
                while split_point != orig_split:
                    invocabs = True
                    for es in embed_sets:
                        if last_utt[split_point] not in vocabs[es]:
                            invocabs = False
                            break
                    if nw_counts[last_utt[split_point]] < MAX_PER_NW and invocabs:
                        break
                    split_point = (split_point + 1) % len(last_utt)

                invocabs = True
                for es in embed_sets:
                    if last_utt[split_point] not in vocabs[es]:
                        invocabs = False
                        break

                if not invocabs or nw_counts[last_utt[split_point]] == MAX_PER_NW:
                    skipped_nw += 1
                    continue
                nw_counts[last_utt[split_point]] += 1
                next_word = {es: vocabs[es][last_utt[split_point]] if last_utt[split_point] in vocabs[es] else vocabs[es][SPTOK_UNK] for es in embed_sets}
                last_utt_split = last_utt[:split_point]
                for token in range(0, len(last_utt_split)):
                    tu_tok += 1
                    try:
                        for es in embed_sets:
                            truncated_utt[es].append(vocabs[es][last_utt_split[token]])
                    except KeyError:
                        for es in embed_sets:
                            truncated_utt[es].append(vocabs[es][SPTOK_UNK])
                        unk_count += 1

            # LIWC feature vector
            liwc_values = liwc_words.tolist()
            liwc_values.extend(liwc_me.tolist())
            liwc_values.extend(liwc_other.tolist())
            for msg_i in feature_window:
                liwc_values.extend(msg_i[3] if msg_i != None else [0]*len(data_set[point][-1][3]))
            if np.linalg.norm(liwc_me) > 0 and np.linalg.norm(liwc_other) > 0:
                liwc_values.append(float(np.dot(liwc_me, liwc_other) / (np.linalg.norm(liwc_me) * np.linalg.norm(liwc_other))))
            else:
                liwc_values.append(0)

            # Response time feature vector
            #response_time = 0 if response_time == None else (last_date - response_time).seconds + (last_date - response_time).days*24*60*60
            first_date = 0 if first_date == None else (last_date - first_date).seconds + (last_date - first_date).days*24*60*60

            # Calculate time value features
            time_values = [first_date, last_date.year-2000, last_date.month, last_date.day, last_date.hour + last_date.minute*1.0/60, month_to_season[last_date.month], math.log(first_date if first_date > 0 else 1)]
            time_values.extend(message_diffs)
            assert (len(message_diffs) == 4 and task != 'ua') or (len(message_diffs) == 5 and task == 'ua')
            for msg_diff in message_diffs:
                time_values.append(math.log(msg_diff if msg_diff > 0 else 1))

            # Calculate frequency value features
            freq_values = [0 if _f == None else _f for _f in freqs]
            freq_diffs = [0 if freqs_end[_i] == None else freqs_end[_i] - freq_values[_i] for _i in range(0, len(freqs))]
            freq_values.extend(freq_diffs[1:]) # the first value is always zero because it is alltime_0 - alltime_ctxlen = ctxlen-1

            # Turn change vector
            tc_values = [1 if _tc != None and _tc[7] else 0 for _tc in feature_window]
            if (len(tc_values) != 4 and task != 'ua') or (len(tc_values) != 5 and task == 'ua'):
                print('tc values: ' + str(tc_values))

            # Stopword counts vector
            stop_values = [_stop[8] if _stop != None else 0 for _stop in feature_window]
            stop_values.extend([stop_me, stop_other])

            # LSM feature vector
            lsm_values = [_lsm[9] if _lsm != None else 0 for _lsm in feature_window]
            lsm_values.append(lsm_values[0]-lsm_values[-1])

            # Surface feature vector
            surface_values = [ctx_words, ctx_me, ctx_other, ctx_me*1.0/ctx_words if ctx_words > 0 else 0, ctx_other*1.0/ctx_words if ctx_words > 0 else 0, sym_all, sym_me, sym_other, sym_me*1.0/sym_all if sym_all > 0 else 0, sym_other*1.0/sym_all if sym_all > 0 else 0, utt_me, utt_other, sym_other/utt_other if utt_other > 0 else 0, sym_me/utt_me if utt_me > 0 else 0]

            # Switch the label basted on the prediction task
            if task == 'cu':
                tlb = data_set[point][-1][2].decode()
            elif task == 'ag':
                # Full utterance for seq2seq
                tlb = data_set[point][-1][2].decode()
            elif task == 'nw':
                # Only take next word for next word prediction
                tlb = next_word
            elif task == 'rt':
                tlb = convert_prt(data_set[point][-1][5])
            elif task == 'ua':
                tlb = data_set[point][-1][4]

            # Lengths should be the same because they use the same tokenization
            dialog_lengths.append(len(dialog[embed_sets[0]]))
            data_set[point] = {'dialog': dialog, 'trunc_utt': truncated_utt, 'user_id': data_set[point][-1][10], 'split_point': split_point, 'label': tlb, 'user_vec': data_set[point][-1][4], 'liwc_words': liwc_values, 'time_values': time_values, 'freq_values': freq_values, 'tc_values': tc_values, 'stop_values': stop_values, 'lsm_values': lsm_values, 'surface_values': surface_values, 'mentions': mention_sum}
            chosen_points.append(point)

            if norm == None:
                user_vecs.append(data_set[point]['user_vec'])
                liwc_vecs.append(data_set[point]['liwc_words'])
                time_vecs.append(data_set[point]['time_values'])
                freq_vecs.append(data_set[point]['freq_values'])
                turn_vecs.append(data_set[point]['tc_values'])
                stop_vecs.append(data_set[point]['stop_values'])
                lsm_vecs.append(data_set[point]['lsm_values'])
                surf_vecs.append(data_set[point]['surface_values'])

        print('Total unknowns: ' + str(unk_count))
        print('Skipped next word instances: ' + str(skipped_nw))
        print('Average unknowns per utterance: ' + str(unk_count*1.0/len(data_set)))
        print('Percent of tokens unknown: ' + str(unk_count*100.0/tu_tok))
        print('Length of dialog contexts avg (stddev): ' + str(np.average(dialog_lengths)) + ' (' + str(np.std(dialog_lengths)) + ')')

    # normalize features
    if norm == None:
        print('Calculating means and stds...')
        user_mean, user_std = np.mean(user_vecs, axis=0), np.std(user_vecs, axis=0)
        liwc_mean, liwc_std = np.mean(liwc_vecs, axis=0), np.std(liwc_vecs, axis=0)
        time_mean, time_std = np.mean(time_vecs, axis=0), np.std(time_vecs, axis=0)
        freq_mean, freq_std = np.mean(freq_vecs, axis=0), np.std(freq_vecs, axis=0)
        turn_mean, turn_std = np.mean(turn_vecs, axis=0), np.std(turn_vecs, axis=0)
        stop_mean, stop_std = np.mean(stop_vecs, axis=0), np.std(stop_vecs, axis=0)
        lsm_mean, lsm_std = np.mean(lsm_vecs, axis=0), np.std(lsm_vecs, axis=0)
        surf_mean, surf_std = np.mean(surf_vecs, axis=0), np.std(surf_vecs, axis=0)
        print('\nUser mean: ' + str(user_mean))
        print('User standard deviation: ' + str(user_std))
        print('\nLIWC mean: ' + str(liwc_mean))
        print('LIWC standard deviation: ' + str(liwc_std))
        print('\nTime mean: ' + str(time_mean))
        print('Time standard deviation: ' + str(time_std))
        print('\nFrequency mean: ' + str(freq_mean))
        print('Frequency standard deviation: ' + str(freq_std))
        print('\nTurn change mean: ' + str(turn_mean))
        print('Turn change standard deviation: ' + str(turn_std))
        print('\nStopword mean: ' + str(stop_mean))
        print('Stopword standard deviation: ' + str(stop_std))
        print('\nLinguistic style matching mean: ' + str(lsm_mean))
        print('Linguistic style matching standard deviation: ' + str(lsm_std))
        print('\nSurface mean: ' + str(surf_mean))
        print('Surface standard deviation: ' + str(surf_std))
    else:
        print('Using precalculated means and stds...')

    print('Replacing values with mean centered and normalized values...')
    for point in range(0, len(data_set)):
        if point in chosen_points:
            data_set[point]['user_vec'] = ((data_set[point]['user_vec'] - user_mean) / user_std).tolist()
            data_set[point]['liwc_words'] = ((data_set[point]['liwc_words'] - liwc_mean) / liwc_std).tolist()
            data_set[point]['time_values'] = ((data_set[point]['time_values'] - time_mean) / time_std).tolist()
            data_set[point]['freq_values'] = ((data_set[point]['freq_values'] - freq_mean) / freq_std).tolist()
            data_set[point]['tc_values'] = ((data_set[point]['tc_values'] - turn_mean) / turn_std).tolist()
            data_set[point]['stop_values'] = ((data_set[point]['stop_values'] - stop_mean) / stop_std).tolist()
            data_set[point]['lsm_values'] = ((data_set[point]['lsm_values'] - lsm_mean) / lsm_std).tolist()
            data_set[point]['surface_values'] = ((data_set[point]['surface_values'] - surf_mean) / surf_std).tolist()

    print('Trimming points...')
    nc_points = [i for i in range(0, len(data_set)) if i not in chosen_points]
    for point in tqdm(sorted(nc_points, reverse=True)):
        del data_set[point]

    # write dataset to file
    f = open_for_write(fname + '/' + section + '_contexts', binary=True)
    f.write(msgpack.packb(data_set))
    f.close()

    return (user_mean, liwc_mean, time_mean, freq_mean, turn_mean, stop_mean, lsm_mean, surf_mean), (user_std, liwc_std, time_std, freq_std, turn_std, stop_std, lsm_std, surf_std)

if __name__ == '__main__':
    main()
