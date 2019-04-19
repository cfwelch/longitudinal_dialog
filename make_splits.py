

import torch, operator, msgpack, nltk, math, os, copy, pytz, sys, re
import dateutil.parser
import scipy.stats
import visdom
import numpy as np

from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
from nltk.corpus import stopwords
from datetime import datetime
from random import random, shuffle
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from utils import settings, label_set, open_for_write, SPTOK_ME, SPTOK_NU, SPTOK_SOS, SPTOK_EOS, SPTOK_OTHER, SPTOK_UNK, SPTOK_NL, EMBEDDING_SIZE, EMBEDDING_LOCATION, DEFAULT_TIMEZONE, time_labels, time_label_size, START_TIME, make_user_vector
from normalizer import expand_text, norms
from user_annotator import read_people_file, tag_set
from mention_graph import read_mention_names

CONTEXT_LEN = 5
OTHER_SAMPLE_RATE = 0.99
NW_SAMPLE_RATE = 0.0
NW_MAXES = [150000, 10000, 10000]
UA_SAMPLE_RATE = 0.99
AIM_FOR_PP = 250
SPLIT_TRAIN, SPLIT_DEV, SPLIT_TEST = [0, 1, 2]
WINDOW_SIZE = 50

vis = visdom.Visdom()

def convert_counts(task, split):
    # initalize
    temp_counts = None
    if task == 'cu':
        temp_counts = {key: 0 for key in label_set}
        temp_counts['other'] = 0
    else:
        temp_counts = {key: 0 for key in time_labels}
    # convert
    for context in split:
        if task == 'rt':
            # for <> median
            #temp_counts['short' if context[-1][5] <= my_median else 'long'] += 1
            temp_counts[convert_prt(context[-1][5])] += 1
        else:
            msg_text = context[-1][2].decode()
            temp_counts[msg_text if msg_text in label_set else 'other'] += 1
    return temp_counts

def convert_prt(point):
    minv, mink = [time_label_size['longer'], 'longer']
    for key in time_label_size:
        if time_label_size[key] > point and time_label_size[key] < minv:
            minv = time_label_size[key]
            mink = key
    return mink

def view_splits(graph_it, task, username, my_median=72):
    fsave_name = '_'.join(username[0].lower().split(' '))
    train, dev, test = [None]*3
    with open('splits/' + settings['prefix'] + '/' + fsave_name + '/train_contexts', 'rb') as handle:
        train = msgpack.unpackb(handle.read())
    with open('splits/' + settings['prefix'] + '/' + fsave_name + '/dev_contexts', 'rb') as handle:
        dev = msgpack.unpackb(handle.read())
    with open('splits/' + settings['prefix'] + '/' + fsave_name + '/test_contexts', 'rb') as handle:
        test = msgpack.unpackb(handle.read())
    print('Length of train contexts: ' + str(len(train)))
    print('Length of dev contexts: ' + str(len(dev)))
    print('Length of test contexts: ' + str(len(test)))

    train_counts = convert_counts(task, train)
    print('Majority class of train on \'' + max(train_counts.keys(), key=lambda x:train_counts[x]) + '\' with ' + '{:2.2f}'.format(max(train_counts.values())*100.0/sum(train_counts.values())) + '%')

    dev_counts = convert_counts(task, dev)
    print('Majority class of development on \'' + max(dev_counts.keys(), key=lambda x:dev_counts[x]) + '\' with ' + '{:2.2f}'.format(max(dev_counts.values())*100.0/sum(dev_counts.values())) + '%')

    test_counts = convert_counts(task, test)
    print('Majority class of test on \'' + max(test_counts.keys(), key=lambda x:test_counts[x]) + '\' with ' + '{:2.2f}'.format(max(test_counts.values())*100.0/sum(test_counts.values())) + '%')

    total_counts = {key: train_counts[key] + dev_counts[key] + test_counts[key] for key in train_counts.keys()}
    if task == 'cu':
        total_counts['other'] = train_counts['other'] + dev_counts['other'] + test_counts['other']
    print('Majority class of total on \'' + max(total_counts.keys(), key=lambda x:total_counts[x]) + '\' with ' + '{:2.2f}'.format(max(total_counts.values())*100.0/sum(total_counts.values())) + '%')
    if graph_it:
        graph_splits(total_counts, train_counts, dev_counts, test_counts, add_name=username[0]+' ')

def make_bar_graph(split, split_name, mbot, qk):
    total_keys = list(split.keys())
    total_keys.sort()
    sort_keys, sort_values = zip(*sorted(split.items(), key=operator.itemgetter(1)))
    env_name = 'Conversation Modeling'
    vis.bar(sort_values, sort_keys, opts=dict(stacked=True, marginbottom=mbot, width=600, title=split_name + ' DA Distribution'), env=env_name)
    print(split_name + ' KL-divergence from total: ' + str(scipy.stats.entropy([split[k] for k in total_keys], qk=qk)))
    vis.save([env_name])

def graph_splits(total, train, dev, test, mbot=70, add_name=''):
    total_keys = list(total.keys())
    total_keys.sort()
    total_dist = [total[k] for k in total_keys]
    make_bar_graph(total, ('Total ' + add_name).strip(), mbot, total_dist)
    make_bar_graph(train, ('Train ' + add_name).strip(), mbot, total_dist)
    make_bar_graph(dev, ('Development ' + add_name).strip(), mbot, total_dist)
    make_bar_graph(test, ('Test ' + add_name).strip(), mbot, total_dist)

def get_embeds(vocab_in, occurs, username, pretrained_path=EMBEDDING_LOCATION + 'glove.840B.300d.txt', is_gensim_model=False, prefix=''):
    fsave_name = '_'.join(username[0].lower().split(' '))
    vocab, re_vocab = OrderedDict(), OrderedDict()
    #with open('splits/' + settings['prefix'] + '/vocabulary_raw', 'r') as f:
    #    vocab_list = f.readlines()
    #vocab_list = [item[:-1] if item.endswith('\n') else item for item in vocab_list]
    vocab_list = list(vocab_in)

    total_word_count = len(vocab_list)
    print('Vocabulary size is: ' + str(total_word_count))
    total_occurs = 0
    ind = 1

    word2vec = {}
    embeddings = []
    # append zero vector
    embeddings.append(torch.zeros(EMBEDDING_SIZE))
    if not is_gensim_model:
        print('Loading pretrained embeddings...')
        with open(pretrained_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                word = ' '.join(values[:len(values)-EMBEDDING_SIZE])
                coefs = np.asarray(values[-EMBEDDING_SIZE:], dtype='float32')
                word2vec[word] = coefs
    else:
        print('Loading pretrained gensim model...')
        model = Word2Vec.load(pretrained_path)
        word2vec = model.wv

    for word in vocab_list:
        if word in word2vec:
            vocab[word] = ind
            re_vocab[ind] = word
            embeddings.append(torch.from_numpy(word2vec[word]))
            ind += 1
            total_occurs += occurs[word]
            assert len(embeddings) == len(vocab) + 1
        # else:
        #     print('word: ' + word + ' is not in w2v')

    # add special tokens
    vocab[SPTOK_ME] = len(vocab)+1
    re_vocab[len(vocab)] = SPTOK_ME
    vocab[SPTOK_OTHER] = len(vocab)+1
    re_vocab[len(vocab)] = SPTOK_OTHER
    vocab[SPTOK_NU] = len(vocab)+1
    re_vocab[len(vocab)] = SPTOK_NU
    vocab[SPTOK_UNK] = len(vocab)+1
    re_vocab[len(vocab)] = SPTOK_UNK
    vocab[SPTOK_NL] = len(vocab)+1
    re_vocab[len(vocab)] = SPTOK_NL

    vocab['<link>'] = len(vocab)+1
    re_vocab[len(vocab)] = '<link>'
    vocab['<picture>'] = len(vocab)+1
    re_vocab[len(vocab)] = '<picture>'

    vocab[SPTOK_SOS] = len(vocab)+1
    re_vocab[len(vocab)] = SPTOK_SOS
    vocab[SPTOK_EOS] = len(vocab)+1
    re_vocab[len(vocab)] = SPTOK_EOS

    print('Length of Vocab: ' + str(len(vocab)))

    # append special token vectors
    for _ in range(9):
        embeddings.append(torch.randn(EMBEDDING_SIZE))

    print('Percentage of unique words with pretrained embeds - ' + str(ind *100.0 / float(total_word_count)) + '%')
    print('Percentage of total words with pretrained embeds - ' + str(total_occurs *100.0 / float(sum(occurs.values()))) + '%')

    embeddings = torch.stack(embeddings)
    print('Saving embeddings to splits folder...')
    torch.save(embeddings, 'splits/' + settings['prefix'] + '/' + fsave_name + '/' + prefix + 'embeds.th')

    handle = open_for_write('splits/' + settings['prefix'] + '/' + fsave_name + '/' + prefix + 'vocabulary', binary=True)
    handle.write(msgpack.packb(vocab))
    handle.close()

    # TODO REMOVE: I think this is pretty useless and in some cases manually flipped the dictionary in memory
    # handle = open_for_write('splits/' + settings['prefix'] + '/' + fsave_name + '/' + prefix + 'reverse_vocabulary', binary=True)
    # handle.write(msgpack.packb(re_vocab))
    # handle.close()
    print('Saved updated vocabulary and reverse-vocabulary to splits folder...')

def generate_splits(task, username, generate):
    fsave_name = '_'.join(username[0].lower().split(' '))
    os.makedirs('splits/' + settings['prefix'] + '/' + fsave_name, exist_ok=True)
    valid_people = read_people_file('tagged_people')
    # so you have a unique index per name
    ppl_map, first_names = read_mention_names()

    people = {}
    vocab = set()
    occurs = defaultdict(lambda: 0)

    zzz = 0
    print('Reading conversation files...')
    for filename in sorted(os.listdir(settings['DATA_MERGE_DIR'])):
        convo = None
        with open(settings['DATA_MERGE_DIR'] + '/' + filename, 'rb') as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b'with'].decode()

        # If the name is not your name then you only have one file to look at!
        if type(username) != list:
            if username not in settings['my_name'] and cname != username:
                continue

        print('File: ' + filename)
        if cname not in valid_people:
            print(cname + ' is not in the list of tagged people...')
            continue
        if cname not in people:
            people[cname] = []

        user_vec = make_user_vector(valid_people, cname)
        for message in tqdm(convo[b'messages']):
            if str(message[b'date']) < str(START_TIME):
                continue
            if b'text' not in message:
                empty_messages += 1
                continue
            if task == 'cu':
                msg_text = message[b'merged']
            else:
                msg_text = message[b'text']
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()
            #msg_text = msg_text.lower()
            #print('-'*30)
            #print('msg_text: ' + msg_text)

            wtok_text = [_t.lower() for _t in nltk.word_tokenize(msg_text)]
            for _tok in wtok_text:
                vocab.add(_tok)
                occurs[_tok] += 1

            # get message date
            mdate = dateutil.parser.parse(message[b'date'])
            #if mdate.tzinfo == None:
            #    mdate = DEFAULT_TIMEZONE.localize(mdate)
            #mmstr = str(((mdate.month-1) // month_bin) * month_bin + 1)
            #mdate = str(mdate.year) + ':' + ('0' + mmstr if len(mmstr) < 2 else mmstr)
            #print('liwc_counts: ' + str(message[b'liwc_counts']))
            #print('user_vec: ' + str(user_vec))

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
            # 10: Index of other speaker in list of speakers ordered by frequency of communication
            # 11: Mention graph counts (b'mentions')
            #########################
            if type(username) == list:
                is_me = message[b'user'].decode() in username
            else:
                is_me = message[b'user'].decode() == username
            people[cname].append([mdate, SPTOK_ME if is_me else SPTOK_OTHER, msg_text, message[b'liwc_counts'], user_vec, message[b'response_time'], [message[b'all_freq'], message[b'month_freq'], message[b'week_freq'], message[b'day_freq']], message[b'turn_change'], message[b'stopword_count'], message[b'lsm'], ppl_map.index(cname), message[b'mentions']])
        zzz += 1
        #if zzz > 5:
        #    break

    if not generate:
        return vocab, occurs

    print('\nOutput conversation lengths: ')
    for k,v in sorted(people.items(), key=lambda i:len(i), reverse=True):
        print(str(k) + '\t' + str(len(v)))

    train_contexts, dev_contexts, test_contexts = [], [], []
    for person in people:
        context = [None] * CONTEXT_LEN
        the_split = None
        win_count = 0
        per_person = 0
        for msg_i in range(0, len(people[person])):
            context[:-1] = context[1:]
            current_msg = people[person][msg_i]
            context[CONTEXT_LEN-1] = current_msg
            the_split = split_probs()

            # Copy context to split -- some redundancy but flexible for future editing
            if (task == 'rt' and current_msg[1] == SPTOK_ME and None not in context) \
            or (task == 'ag' and current_msg[1] == SPTOK_ME and None not in context) \
            or (task == 'nw' and current_msg[1] == SPTOK_ME and None not in context and random() > NW_SAMPLE_RATE) \
            or (task == 'cu' and (current_msg[2] in label_set or random() > OTHER_SAMPLE_RATE) and current_msg[1] == SPTOK_ME) \
            or (task == 'ua' and (random() > UA_SAMPLE_RATE or per_person < AIM_FOR_PP)):

                tcon = copy.deepcopy(context)
                #print(tcon)
                for tmsg in tcon:
                    if tmsg != None:
                        tmsg[0] = tmsg[0].isoformat()

                if the_split == SPLIT_TRAIN:
                    train_contexts.append(tcon)
                elif the_split == SPLIT_DEV:
                    dev_contexts.append(tcon)
                elif the_split == SPLIT_TEST:
                    test_contexts.append(tcon)
                else:
                    raise

                if win_count > WINDOW_SIZE:
                    context = [None] * CONTEXT_LEN
                    the_split = split_probs()
                    win_count = 0

                per_person += 1
        # if the task is user attributes, then make sure you have points from all ppl
        assert task != 'ua' or per_person > 0

    if task == 'nw':
        shuffle(train_contexts)
        train_contexts = train_contexts[:NW_MAXES[0]]
        shuffle(dev_contexts)
        dev_contexts = dev_contexts[:NW_MAXES[1]]
        shuffle(test_contexts)
        test_contexts = test_contexts[:NW_MAXES[2]]

    print('Train contexts size: ' + str(len(train_contexts)))
    handle = open_for_write('splits/' + settings['prefix'] + '/' + fsave_name + '/train_contexts', binary=True)
    handle.write(msgpack.packb(train_contexts))
    handle.close()

    print('Development contexts size: ' + str(len(dev_contexts)))
    handle = open_for_write('splits/' + settings['prefix'] + '/' + fsave_name + '/dev_contexts', binary=True)
    handle.write(msgpack.packb(dev_contexts))
    handle.close()

    print('Test contexts size: ' + str(len(test_contexts)))
    handle = open_for_write('splits/' + settings['prefix'] + '/' + fsave_name + '/test_contexts', binary=True)
    handle.write(msgpack.packb(test_contexts))
    handle.close()

    #print('Writing vocab file...')
    #with open('splits/' + settings['prefix'] + '/vocabulary_raw', 'w') as f:
    #    f.write('\n'.join(list(vocab)))
    if task == 'ua':
        print('Sum of lengths of context files: ' + str(sum([len(test_contexts), len(train_contexts), len(dev_contexts)])))
    return vocab, occurs

def split_probs():
    the_split = None
    if random() < 0.8:
        the_split = SPLIT_TRAIN
    else:
        if random() > 0.5:
            the_split = SPLIT_DEV
        else:
            the_split = SPLIT_TEST
    return the_split

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-g', '--gen', dest='gen', help='Do not generate by default', default=False, action='store_true')
    parser.add_argument('-e', '--embeddings', dest='embeddings', help='Do not generate embeddings subset file by default but do if generating anything', default=False, action='store_true')
    parser.add_argument('-s', '--show', dest='graph_it', help='Do not graph it by default', default=False, action='store_true')
    parser.add_argument('-t', '--task', dest='task', help='Problem formulation is response time (rt), user attribute (ua), common utterance classification (cu), all utterance generation (ag which is for seq2seq), or next word prediction (nw)', default='', type=str)
    parser.add_argument('-u', '--user', dest='user', help='Decide which user should be used for prediction. Input their name. Default is settings[\'my_name\'] from utils.py', default=settings['my_name'], type=str)
    opt = parser.parse_args()

    if opt.task == '':
        print('Select response time (rt), user attributes (ua), or common utterance classification (cu).')
        sys.exit(0)

    if opt.gen:
        print('Generating split with user \'' + str(opt.user) + '\'...')
        vocab, occurs = generate_splits(opt.task, opt.user, True)
        get_embeds(vocab, occurs, opt.user, pretrained_path=EMBEDDING_LOCATION+'glove.840B.300d.txt', is_gensim_model=False)

    if opt.embeddings and not opt.gen:
        print('Getting voacbulary...')
        vocab, occurs = generate_splits(opt.task, opt.user, False)
        get_embeds(vocab, occurs, opt.user, pretrained_path=EMBEDDING_LOCATION+'glove.840B.300d.txt', is_gensim_model=False)

    if opt.task not in ['ua', 'nw']:
        view_splits(opt.graph_it, opt.task, opt.user)
