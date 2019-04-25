

import datetime, inspect, random, msgpack, os, sys, math, time, torch, pdb
import numpy as np

from pprint import pprint
from scipy.sparse.csgraph import floyd_warshall

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import gprint, label_set, time_labels, SPTOK_ME, SPTOK_NU, SPTOK_OTHER, SPTOK_UNK, SPTOK_EOS, tag_set, RA_IND, FAM_IND, SCC_IND, SG_IND, SCH_IND, WORK_IND, NPR_IND, uv_to_parts, uax_set, NUMBER_OF_PEOPLE

def cprint(msg, error=False, important=False):
    gprint(msg, "dataloader", error, important)

PERSON_BATCH_SIZE = 1

class Dataloader():
    def __init__(self, epochs, batch_size, cont, rt, ua, start_split, single_att, vocab=None, next_word=False, embed_prefix=None, max_length=-1):
        self.nbatch_returns = 16
        self.train_data_path = "train_contexts"
        self.dev_data_path = "dev_contexts"
        self.test_data_path = "test_contexts"

        cprint("\tTrain File Path: " + self.train_data_path)
        cprint("\tTest File Path: " + self.test_data_path)
        cprint("\tDevelopment File Path: " + self.dev_data_path)

        assert os.path.isfile(self.train_data_path) and os.path.isfile(self.dev_data_path) and os.path.isfile(self.test_data_path), 'Given train/dev/test data files do not exist at path'

        self.max_epochs = epochs
        self.batch_size = batch_size
        self.iter_ind = 1
        self.global_ind = 1
        self.stop_flag = False
        self.start_split = start_split
        self.single_att = single_att
        self.vocab = vocab
        self.ppl_key_set = None
        self.next_word = next_word
        self.embed_prefix = embed_prefix
        self.max_length = max_length

        self.is_cont = cont
        self.is_rt = rt
        self.is_ua = ua
        self.people_map = {}
        self.serving_person = start_split
        self.ppl_keys = None

        self.liwc_len = 0
        self.time_len = 0
        self.freq_len = 0
        self.tc_len = 0
        self.stop_len = 0
        self.lsm_len = 0
        self.surf_len = 0

        self.all_data_sum = 0

    # Run this once before training to load the required dataset split from disk
    def serve_data(self, all_data=False):
        self.iter_ind = 1
        self.stop_flag = False

        if not os.path.isfile(self.train_data_path) and not os.path.isfile(self.dev_data_path):
            cprint("Cannot find train/dev data OR labels. Generate the splits and re-run this script.", error=True)
            sys.exit(0)

        self.epoch = 0
        self.iter_ind = 0
        self.global_ind = 1
        with open(self.train_data_path, 'rb') as f:
            self.train_data = msgpack.unpackb(f.read())
            # self.train_data = self.train_data[:10000] #HACK
        with open(self.dev_data_path, 'rb') as f:
            self.dev_data = msgpack.unpackb(f.read())
            #self.dev_data = self.dev_data[:1000] #HACK
        self.all_data_sum = len(self.train_data) + len(self.dev_data)

        cprint('%d training conversation files loaded from: %s ' % (len(self.train_data), self.train_data_path))
        cprint('%d training conversation files loaded from: %s ' % (len(self.dev_data), self.dev_data_path))

        # Load test data
        with open(self.test_data_path, 'rb') as f:
            self.test_data = msgpack.unpackb(f.read())
        cprint('%d test conversation files loaded from: %s ' % (len(self.test_data), self.test_data_path))
        if all_data:
            # Merge train and development splits
            self.train_data.extend(self.dev_data)
            self.all_data_sum += len(self.test_data)
            if self.is_ua:
                self.train_data.extend(self.test_data)
        cprint("Total train samples: " + str(len(self.train_data)))

        ##################################################
        # Needed initalization of people map for user attribute classification but also for using the graph features.
        for ind in range(0, len(self.train_data)):
            p_id = self.train_data[ind][b'user_id']
            if p_id not in self.people_map:
                self.people_map[p_id] = []
            if self.is_ua:
                self.people_map[p_id].append(self.train_data[ind])
        self.ppl_keys = list(self.people_map.keys())
        self.ppl_keys.sort()
        self.train_graph = np.zeros((NUMBER_OF_PEOPLE, NUMBER_OF_PEOPLE))
        self.dev_graph = np.copy(self.train_graph)
        self.test_graph = np.copy(self.train_graph)
        ##################################################

        if self.is_ua:
            cprint("\nUser distribution: ")
            for p_key in self.people_map:
                cprint(str(p_key) + ": " + str(len(self.people_map[p_key])))
            cprint("\n")
            self.next_ua()
        else:
            self.train_inds = list(range(len(self.train_data)))
            random.shuffle(self.train_inds)

        self.liwc_len = len(self.train_data[0][b'liwc_words'])
        self.time_len = len(self.train_data[0][b'time_values'])
        self.freq_len = len(self.train_data[0][b'freq_values'])
        self.tc_len = len(self.train_data[0][b'tc_values'])
        self.stop_len = len(self.train_data[0][b'stop_values'])
        self.lsm_len = len(self.train_data[0][b'lsm_values'])
        self.surf_len = len(self.train_data[0][b'surface_values'])

    # This is for next word prediction weights
    def get_label_weights(self, lenv):
        #MIN_WEIGHT = 0.000005
        weights = [0]*lenv
        for i in range(len(self.train_data)):
            #hack
            ttt = self.train_data[i][b'label'][self.embed_prefix.encode() if self.embed_prefix != None else b'']
            weights[ttt] += 1
        wsum = max(weights) + 1 #sum(weights)
        for i in range(len(weights)):
            #weights[i] = 1.0 - weights[i] * 1.0 / wsum
            weights[i] = 1.0 - math.pow(weights[i] * 1.0 / wsum, 10)
        return weights

    def next_ua(self):
        # reset flags
        self.iter_ind = 0
        self.epoch = 0
        self.global_ind = 1
        self.stop_flag = False

        # resplit data
        self.train_data = []
        self.dev_data = []

        adder = self.serving_person + PERSON_BATCH_SIZE
        if adder + PERSON_BATCH_SIZE > len(self.ppl_keys):
            self.ppl_key_set = self.ppl_keys[self.serving_person:]
        else:
            self.ppl_key_set = self.ppl_keys[self.serving_person:adder]

        self.test_data = {pk_: [] for pk_ in self.ppl_key_set}

        for key in self.people_map:
            for ind in range(0, len(self.people_map[key])):
                if key in self.ppl_key_set:
                    self.test_data[key].append(self.people_map[key][ind])
                elif random.random() > 0.90:
                    self.dev_data.append(self.people_map[key][ind])
                else:
                    self.train_data.append(self.people_map[key][ind])

        assert len(self.train_data) + len(self.dev_data) + sum([len(self.test_data[pksl]) for pksl in self.test_data]) == self.all_data_sum

        self.train_inds = list(range(len(self.train_data)))
        random.shuffle(self.train_inds)

        # regenerate train distance graph
        self.train_graph = np.zeros((len(self.ppl_keys), len(self.ppl_keys)))
        for t_point in self.train_data:
            self.train_graph[t_point[b'user_id']] += t_point[b'mentions']
        #print('The train graph weights:')
        #for t_point in self.train_graph:
        #    print('\t' + str(t_point.tolist()))

        self.dev_graph = np.copy(self.train_graph)
        for t_point in self.dev_data:
            self.dev_graph[t_point[b'user_id']] += t_point[b'mentions']
        #print('The dev graph weights:')
        #for t_point in self.dev_graph:
        #    print('\t' + str(t_point.tolist()))

        self.test_graph = np.copy(self.dev_graph)
        for key in self.ppl_key_set:
            for t_point in self.test_data[key]:
                self.test_graph[t_point[b'user_id']] += t_point[b'mentions']
        #print('The test graph weights:')
        #for t_point in self.test_graph:
        #    print('\t' + str(t_point.tolist()))

        self.get_graph_dists(self.train_graph)
        self.get_graph_dists(self.dev_graph)
        self.get_graph_dists(self.test_graph)

    def get_graph_dists(self, graph_type):
        train_max = np.max(graph_type)
        train_min = np.min(graph_type)
        for t_gx in range(len(self.ppl_keys)):
            for t_gy in range(len(self.ppl_keys)):
                graph_type[t_gx][t_gy] = 1 - (train_max - graph_type[t_gx][t_gy]) * 1.0 / (train_max - train_min)
        graph_type = floyd_warshall(graph_type)
        graph_type = np.where(graph_type != np.inf, graph_type, 100.0)
        #print('The graph distances:')
        #for t_point in graph_type:
        #    print('\t' + str(t_point.tolist()))

    def get_train_batch(self):
        Data = self.train_data
        if self.epoch > self.max_epochs:
            cprint("Maximum Epoch Limit reached")
            self.stop_flag = True
            return [None] * self.nbatch_returns
        # If the iterator is at the end of the samples
        # If fewer samples left in epoch than batch size
        if self.iter_ind + self.batch_size - 1 > len(self.train_inds):
            subsample = self.train_inds[self.iter_ind:]
        else:
        # If enough samples for minibatch; get indices
            subsample = self.train_inds[self.iter_ind:self.iter_ind+self.batch_size]
        return self.get_batch(Data, subsample, self.train_graph, is_training=True)

    def get_evaluation_data(self, split):
        assert split == 'test' or split == 'dev', 'Split parameter has to be test or dev...'
        if split == 'test':
            params_set = []
            if self.is_ua:
                for key in self.ppl_key_set:
                    Data = self.test_data[key]
                    subsample = range(len(Data))
                    cprint("Using " + str(len(Data)) + " (all) samples in " + split + " split...")
                    params = []
                    for i in subsample:
                        params.append(self.get_batch([Data[i]], [0], self.test_graph))
                    params_set.append(params)
            else:
                Data = self.test_data
                subsample = range(len(Data))
                cprint("Using " + str(len(Data)) + " (all) samples in " + split + " split...")
                params = []
                for i in subsample:
                    params.append(self.get_batch([Data[i]], [0], self.test_graph))
                params_set.append(params)
            return params_set
        else:
            Data = self.dev_data
            subsample = range(len(Data))
            cprint("Using " + str(len(Data)) + " (all) samples in " + split + " split...")
            params = []
            for i in subsample:
                params.append(self.get_batch([Data[i]], [0], self.dev_graph))
            return params

    def get_batch(self, batch_data, subsample, the_graph, is_training=False):
        utter_batch, rev_batch, labels, lens, user_rep, liwc_values, time_values, freq_values, tc_values, stop_values, lsm_values, surface_values, graph_values, oso_batch, oso_rev_batch, oso_lens = [[] for i in range(self.nbatch_returns)]
        max_len, max_oso_len = [0, 0]
        ep = self.embed_prefix.encode() if self.embed_prefix != None else b''

        if self.is_ua:
            fam_labels, npr_labels, ra_labels, scc_labels, sg_labels, sch_labels, work_labels = [[] for i in range(7)]
        # Loop over samples
        for ind in subsample:
            sample = batch_data[ind]
            # array has CONTEXT_LEN dicts of {'dialog', 'label', 'user_vec'}
            if self.next_word:
                label = sample[b'label'][ep]

                if self.max_length > 0:
                    if len(sample[b'dialog'][ep]) > self.max_length-1:
                        sample[b'dialog'][ep] = sample[b'dialog'][ep][:self.max_length-1]
                    else:
                        sample[b'dialog'][ep].extend([self.vocab[SPTOK_EOS]] * (self.max_length - len(sample[b'dialog'][ep]) - 1))

                sample[b'dialog'][ep].append(self.vocab[SPTOK_EOS])
                utter_batch.append(sample[b'dialog'][ep])
                rev_batch.append(list(reversed(sample[b'dialog'][ep])))
                oso_batch.append(sample[b'trunc_utt'][ep])
                oso_rev_batch.append(list(reversed(sample[b'trunc_utt'][ep])))
            else:
                label = sample[b'label'].decode() if not self.is_cont and not self.is_ua else sample[b'label']
                # do batch stuff and get reversed utterance for bidirectional model
                utter_batch.append(sample[b'dialog'][ep])
                rev_batch.append(list(reversed(sample[b'dialog'][ep])))

            liwc_values.append(torch.FloatTensor(sample[b'liwc_words']))
            time_values.append(torch.FloatTensor(sample[b'time_values']))
            freq_values.append(torch.FloatTensor(sample[b'freq_values']))
            tc_values.append(torch.FloatTensor(sample[b'tc_values']))
            stop_values.append(torch.FloatTensor(sample[b'stop_values']))
            lsm_values.append(torch.FloatTensor(sample[b'lsm_values']))
            surface_values.append(torch.FloatTensor(sample[b'surface_values']))
            graph_values.append(torch.FloatTensor(the_graph[sample[b'user_id']]))

            t_user_rep = np.nan_to_num(sample[b'user_vec'])
            if self.is_ua and self.single_att > -1:
                ktemp = uv_to_parts(t_user_rep)
                ktemp[uax_set[self.single_att]] = [0]*len(ktemp[uax_set[self.single_att]])
                ktem2 = []
                for sub in ktemp:
                    ktem2.extend(sub)
                t_user_rep = ktem2
            user_rep.append(torch.FloatTensor(t_user_rep))

            lens.append(len(utter_batch[-1]) - 1)
            if self.next_word:
                oso_lens.append(len(oso_batch[-1]) - 1)
            if self.is_cont:
                labels.append(label)
            else:
                if self.is_ua:
                    #keys = list(tag_set.keys())
                    #keys.sort()
                    #value_set = tag_set[keys[self.uatt]]
                    val_set = uv_to_parts(label)
                    fam_labels.append(val_set[FAM_IND].index(1))
                    npr_labels.append(val_set[NPR_IND].index(1))
                    ra_labels.append(val_set[RA_IND].index(1))
                    scc_val = val_set[SCC_IND].index(1)
                    scc_labels.append(scc_val if scc_val != 2 else 1)
                    sg_labels.append(val_set[SG_IND].index(1))
                    sch_labels.append(val_set[SCH_IND].index(1))
                    work_labels.append(val_set[WORK_IND].index(1))
                elif self.next_word:
                    labels.append(label)
                else:
                    ga_lbs = label_set if not self.is_rt else time_labels
                    # label is index of word or last index, which is the length, meaning 'other'
                    labels.append(ga_lbs.index(label) if label in ga_lbs else len(ga_lbs))
            #print("Label given is: " + str(labels[-1]) + " for word: " + label)
            max_len = len(utter_batch[-1]) if max_len < len(utter_batch[-1]) else max_len
            if self.next_word:
                max_oso_len = len(oso_batch[-1]) if max_oso_len < len(oso_batch[-1]) else max_oso_len

        # pad with 0 tokens and convert to torch tensors
        rev_batch = [torch.LongTensor(item + [0] * (max_len - len(item))) for item in rev_batch]
        utter_batch = [torch.LongTensor(item + [0] * (max_len - len(item))) for item in utter_batch]
        if self.next_word:
            oso_rev_batch = [torch.LongTensor(item + [0] * (max_oso_len - len(item))) for item in oso_rev_batch]
            oso_batch = [torch.LongTensor(item + [0] * (max_oso_len - len(item))) for item in oso_batch]

        utter_batch = torch.stack(utter_batch, 0)
        rev_batch = torch.stack(rev_batch, 0)
        if self.next_word:
            oso_batch = torch.stack(oso_batch, 0)
            oso_rev_batch = torch.stack(oso_rev_batch, 0)

        user_rep = torch.stack(user_rep, 0)
        liwc_values = torch.stack(liwc_values, 0)
        time_values = torch.stack(time_values, 0)
        freq_values = torch.stack(freq_values, 0)
        tc_values = torch.stack(tc_values, 0)
        stop_values = torch.stack(stop_values, 0)
        lsm_values = torch.stack(lsm_values, 0)
        surface_values = torch.stack(surface_values, 0)
        graph_values = torch.stack(graph_values, 0)

        lens = torch.LongTensor(lens)
        oso_lens = torch.LongTensor(oso_lens)
        #print("Lengths size: " + str(lens.size()))
        if self.is_ua:
            fam_labels = torch.LongTensor(fam_labels)
            npr_labels = torch.LongTensor(npr_labels)
            ra_labels = torch.LongTensor(ra_labels)
            scc_labels = torch.LongTensor(scc_labels)
            sg_labels = torch.LongTensor(sg_labels)
            sch_labels = torch.LongTensor(sch_labels)
            work_labels = torch.LongTensor(work_labels)
            labels = [fam_labels, npr_labels, ra_labels, scc_labels, sg_labels, sch_labels, work_labels]
        else:
            labels = torch.LongTensor(labels) if not self.is_cont else torch.FloatTensor(labels)

        if is_training:
            self.iter_ind += self.batch_size
            if self.iter_ind >= len(self.train_inds):
                self.iter_ind = 0
                self.global_ind = 1
                random.shuffle(self.train_inds)
                self.epoch = self.epoch + 1
            self.global_ind += 1

        #print("Max batch: " + str(utter_batch.max()))
        ret_vals = [utter_batch, rev_batch, labels, lens, user_rep, liwc_values, time_values, freq_values, tc_values, stop_values, lsm_values, surface_values, graph_values]
        if self.next_word:
            ret_vals.extend([oso_batch, oso_rev_batch, oso_lens])
        return ret_vals
