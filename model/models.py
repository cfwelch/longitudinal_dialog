

import torch, pdb, sys, os
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from torch.autograd import Variable

from utils import liwc_keys, tag_set, FAM_IND, NPR_IND, RA_IND, SCC_IND, SG_IND, SCH_IND, WORK_IND

# Seq2Vec multi-class classification for user attributes, response time, and common utterance classification.
class net(nn.Module):
    def __init__(self, hidden_size, output_acts, word_embed, embed_size, p=0.5, add_user_representation=False, add_liwc_counts=False, add_time_values=False, add_freq_values=False, add_tc_values=False, add_stop_values=False, add_lsm_values=False, add_graph_values=False, add_surf_values=False, liwc_len=0, time_len=0, freq_len=0, tc_len=0, stop_len=0, lsm_len=0, surf_len=0, graph_len=0, is_cont=False, is_ua=False, s_ua=-1):
        # Basic initalization
        super(net, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.word_embed = word_embed
        self.main_directions = 2
        self.is_ua = is_ua
        self.s_ua = s_ua
        self.add_user_rep = add_user_representation
        self.add_liwc_counts = add_liwc_counts
        self.add_time_values = add_time_values
        self.add_freq_values = add_freq_values
        self.add_tc_values = add_tc_values
        self.add_stop_values = add_stop_values
        self.add_lsm_values = add_lsm_values
        self.add_graph_values = add_graph_values
        self.add_surf_values = add_surf_values

        self.liwc_len = liwc_len
        self.time_len = time_len
        self.freq_len = freq_len
        self.tc_len = tc_len
        self.stop_len = stop_len
        self.lsm_len = lsm_len
        self.surf_len = surf_len
        self.graph_len = graph_len

        # Normal layers
        feature_sets = sum(1 if feature_set else 0 for feature_set in [self.add_user_rep, self.add_liwc_counts, self.add_time_values, self.add_freq_values, self.add_tc_values, self.add_stop_values, self.add_lsm_values, self.add_surf_values, self.add_graph_values])
        self.fwdlstm = nn.LSTM(self.embed_size, hidden_size, bias=True, dropout=p, batch_first=True)
        self.bkwdlstm = nn.LSTM(self.embed_size, hidden_size, bias=True, dropout=p, batch_first=True)
        #self.out_non_linear = nn.Tanh()
        self.out_dropout = nn.Dropout(p=p)

        # Encoding feature layers
        self.encode_user = nn.Linear(sum([len(tag_set[tag]) for tag in tag_set]), hidden_size)
        self.encode_liwc = nn.Linear(liwc_len, hidden_size)
        self.encode_time = nn.Linear(time_len, hidden_size)
        self.encode_freq = nn.Linear(freq_len, hidden_size)
        self.encode_tc = nn.Linear(tc_len, hidden_size)
        self.encode_stop = nn.Linear(stop_len, hidden_size)
        self.encode_lsm = nn.Linear(lsm_len, hidden_size)
        self.encode_surf = nn.Linear(surf_len, hidden_size)
        self.encode_graph = nn.Linear(graph_len, hidden_size)

        to_init = [self.fwdlstm.weight_hh_l0, self.fwdlstm.weight_ih_l0, self.bkwdlstm.weight_hh_l0, self.bkwdlstm.weight_ih_l0, self.encode_liwc.weight, self.encode_time.weight, self.encode_user.weight, self.encode_freq.weight, self.encode_tc.weight, self.encode_stop.weight, self.encode_lsm.weight, self.encode_surf.weight, self.encode_graph.weight]

        if self.is_ua:
            keys = list(tag_set.keys())
            keys.sort()
            #['family', 'non-platonic relationship', 'relative age', 'same childhood country', 'same gender', 'school', 'shared ethnicity', 'work']
            assert keys[0] == 'family' and keys[1] == 'non-platonic relationship' and keys[2] == 'relative age' and keys[3] == 'same childhood country' and keys[4] == 'same gender' and keys[5] == 'school' and keys[7] == 'work'
            self.fam_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, 2)
            self.npr_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, 2)
            self.ra_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, 3)
            self.scc_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, 2)
            self.sg_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, 2)
            self.sch_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, 2)
            self.work_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, 2)
            # initalize weights
            to_init.extend([self.fam_proj.weight, self.npr_proj.weight, self.ra_proj.weight, self.scc_proj.weight, self.sg_proj.weight, self.sch_proj.weight, self.work_proj.weight])
        else:
            self.out_proj = nn.Linear((self.main_directions + feature_sets) * hidden_size, output_acts if not is_cont else 1)
            # initalize weights
            to_init.append(self.out_proj.weight)

        # Initalize all weights to xaiver uniform
        for layer in to_init:
            layer.data = xavier_uniform_(torch.FloatTensor(layer.data.size()))

    # input_matrix: batch x token x word2vec
    def forward(self, input_matrix, lens, user_rep, liwc_sum, time_values, freq_values, tc_values, stop_values, lsm_values, surf_values, graph_values):
        embed_batch, rev_batch = input_matrix
        batch_size = embed_batch.size(0)
        #batch_length = embed_batch.size(1)

        fword_embed = self.word_embed(embed_batch)
        bword_embed = self.word_embed(rev_batch)

        # Get large LSTM output
        fh, _ = self.fwdlstm(fword_embed)
        bh, _ = self.bkwdlstm(bword_embed)

        ## prepare for gather
        lens = lens.view(batch_size, 1, 1).expand(batch_size, 1, self.hidden_size) # get the lengths of the utterances in the batch
        fwd_last_hidden = fh.gather(1, lens).squeeze(1) # get the LSTM output at the end of the sequence
        bkwd_last_hidden = bh.gather(1, lens).squeeze(1) # get the LSTM output at the end of the backward sequence
        hidd = torch.cat((fwd_last_hidden, bkwd_last_hidden), 1) # concatenate the two LSTM outputs #past self.out_non_linear

        if self.add_user_rep:
            user_encoded = self.encode_user(user_rep)
            hidd = torch.cat((hidd, user_encoded), 1)
        if self.add_liwc_counts:
            liwc_encoded = self.encode_liwc(liwc_sum)
            hidd = torch.cat((hidd, liwc_encoded), 1)
        if self.add_time_values:
            time_encoded = self.encode_time(time_values)
            hidd = torch.cat((hidd, time_encoded), 1)
        if self.add_freq_values:
            freq_encoded = self.encode_freq(freq_values)
            hidd = torch.cat((hidd, freq_encoded), 1)
        if self.add_tc_values:
            tc_encoded = self.encode_tc(tc_values)
            hidd = torch.cat((hidd, tc_encoded), 1)
        if self.add_stop_values:
            stop_encoded = self.encode_stop(stop_values)
            hidd = torch.cat((hidd, stop_encoded), 1)
        if self.add_lsm_values:
            lsm_encoded = self.encode_lsm(lsm_values)
            hidd = torch.cat((hidd, lsm_encoded), 1)
        if self.add_surf_values:
            surf_encoded = self.encode_surf(surf_values)
            hidd = torch.cat((hidd, surf_encoded), 1)
        if self.add_graph_values:
            graph_encoded = self.encode_graph(graph_values)
            hidd = torch.cat((hidd, graph_encoded), 1)

        ## pass the concatenated vector to a linear projection
        # could use out_dropout or non_linear
        if self.is_ua:
            if self.s_ua == -1:
                fam_logits = self.fam_proj(self.out_dropout(hidd))
                npr_logits = self.npr_proj(self.out_dropout(hidd))
                ra_logits = self.ra_proj(self.out_dropout(hidd))
                scc_logits = self.scc_proj(self.out_dropout(hidd))
                sg_logits = self.sg_proj(self.out_dropout(hidd))
                sch_logits = self.sch_proj(self.out_dropout(hidd))
                work_logits = self.work_proj(self.out_dropout(hidd))
                logits = [fam_logits, npr_logits, ra_logits, scc_logits, sg_logits, sch_logits, work_logits]
            #[FAM_IND, NPR_IND, RA_IND, SCC_IND, SG_IND, SCH_IND, WORK_IND]
            elif self.s_ua == 0:
                logits = self.fam_proj(self.out_dropout(hidd))
            elif self.s_ua == 1:
                logits = self.npr_proj(self.out_dropout(hidd))
            elif self.s_ua == 2:
                logits = self.ra_proj(self.out_dropout(hidd))
            elif self.s_ua == 3:
                logits = self.scc_proj(self.out_dropout(hidd))
            elif self.s_ua == 4:
                logits = self.sg_proj(self.out_dropout(hidd))
            elif self.s_ua == 5:
                logits = self.sch_proj(self.out_dropout(hidd))
            elif self.s_ua == 6:
                logits = self.work_proj(self.out_dropout(hidd))
        else:
            logits = self.out_proj(self.out_dropout(hidd))
        return logits
