

import matplotlib.pyplot as plt
import operator, random, msgpack, math, sys, os, re
import numpy as np

from prettytable import PrettyTable
from argparse import ArgumentParser
from textwrap import wrap

from utils import p_fix, tag_set, uax_names, uax_set
from mention_graph import read_mention_names

colors = ['b', 'g', 'r', 'c', 'm', 'y']

def main():
    parser = ArgumentParser()
    #parser.add_argument("-pp", "--per-person", dest="per_person", help="Generate per person statistics.", default=None, type=str)
    parser.add_argument("-joint", "--joint", dest="joint", help="Generate statistics for joint models instead of single attribute.", default=False, action="store_true")
    opt = parser.parse_args()

    #per_p = opt.per_person
    #if opt.per_person != None:
    #    if per_p.startswith('logs/'):
    #        per_p = per_p[5:]
    log_acc(opt.joint)

def log_acc(joint):
    ppl_map, first_names = read_mention_names()
    table = PrettyTable(["Attribute", "Features", "Agg. Acc.", "pPerson Mean", "pPerson Std."])
    table.align["Attribute"] = "l"
    tag_keys = [i for i in tag_set.keys()]
    tag_keys.sort()

    ppl_accs = {}
    log_list = os.listdir('logs')# if log_name == None else [log_name]

    for fname in log_list:
        #if not fname.startswith('single_att') and log_name == None:
        if (not fname.startswith('single_att') and not joint) or (not fname.startswith('model') and joint):
            #print(fname + ' is not a single attribute log, skipping...')
            continue

        fset = []
        the_tag = None
        fset_active, att_next = False, False
        for i in fname[:-4].split('_'):
            if fset_active:
                fset.append(i)
            elif i == 'model':
                fset_active = True
            elif i == 'att':
                att_next = True
            elif att_next:
                att_next = False
                the_tag = i
                the_tag = tag_keys[uax_set[uax_names.index(the_tag)]]
        fset = ','.join(fset)
        print(fname + ' - ' + (fset if '' != fset else 'None'))

        log_lines = None
        with open('logs/' + fname) as handle:
            log_lines = handle.readlines()

        skip_log = True
        for line in log_lines:
            if 'Best parameter set is' in line:
                skip_log = False
                break

        if skip_log:
            print(fname + ' is not complete, skipping...')
        else:
            epoch_acc = -1
            cur_person = -1
            cur_attribute = 'null'
            testing = False

            if fset not in ppl_accs:
                ppl_accs[fset] = {name: {k: v for ttag in tag_set for k, v in zip((ttag, ttag+'_t', ttag+'_rp'), ({k2: v2 for the_val in tag_set[ttag] for k2, v2 in zip((the_val, the_val+'_rp'), (0, [0, 0]))}, 0, [0, 0]))} for name in ppl_map}
            #if log_name != None and log_name.startswith('model'):
            #else:
            #ppl_accs = {name: {k: v for k, v in zip((the_tag, the_tag+'_t'), ({the_val: 0 for the_val in tag_set[the_tag]}, 0))} for name in ppl_map}
            overall_acc = {ttag: 0 for ttag in tag_set}
            in_oacc = False
            label_vs = False

            for line in log_lines:
                tline = line.strip()
                if tline == '':
                    continue
                tline = tline[20:].strip()
                #print(tline)

                if tline.startswith('Guessing that person has') and label_vs:
                    label_vs = False

                if 'END OF EPOCH' in tline:
                    tline = tline.replace('-', '')
                    tline = tline.split(':')[1]
                    epoch_acc = int(tline.strip())
                    #print('epoch is now: ' + str(epoch_acc))
                    testing = False
                    cur_attribute = 'null'
                elif 'Best development accuracy so far. Checking test...' in tline:
                    testing = True
                elif re.match('Person \d+...', tline):
                    temp_person = int(tline[6:-3])
                    if temp_person > cur_person:
                        cur_person = temp_person
                    #print('Current person is now: ' + str(cur_person))
                elif tline.startswith('User Attribute'):
                    cur_attribute = tline[15:]
                elif ' correct out of ' in tline and testing:
                    tline = tline.split(' correct out of ')
                    label_vs = True
                    #print(tline)
                    ppl_accs[fset][ppl_map[cur_person]][cur_attribute+'_t'] = int(tline[0]) * 1.0 / int(tline[1])
                    ppl_accs[fset][ppl_map[cur_person]][cur_attribute+'_rp'] = (float(tline[0]), float(tline[1]))
                elif label_vs:
                    #print('checking a label value...')
                    if tline.startswith('Epoch') or ':' not in tline or '=' not in tline: # hack for mixed logs
                        continue
                    tval = tline.split(":")[0].strip()
                    tacc = tline.split("=")[1].strip()
                    ratio_parts = tline[tline.index(':')+1:]
                    ratio_parts = ratio_parts[:ratio_parts.index('=')].strip()
                    ratio_parts = [float(rp) for rp in ratio_parts.split('/')]
                    #print('rp: ' + str(ratio_parts))
                    ppl_accs[fset][ppl_map[cur_person]][cur_attribute][tval] = float(tacc)
                    ppl_accs[fset][ppl_map[cur_person]][cur_attribute][tval+'_rp'] = ratio_parts
                    #print("tval (" + str(tval) + ") has acc " + str(tacc))
                elif '*******************************' in tline:
                    in_oacc = False
                elif 'Accuracy so far' in tline:
                    in_oacc = True
                elif in_oacc and '/104 =' in tline:
                    tline = tline.split(' = ')
                    colpart = tline[0].split(':')
                    if len(colpart) > 1 and (colpart[0].strip() == the_tag or joint):
                        overall_acc[colpart[0].strip()] = tline[1]

            taglist = [ttag for ttag in tag_set] if joint else [the_tag]
            for iter_tag in taglist:
                print('-'*30)
                print(iter_tag)

                # add to pretty table
                table.add_row([iter_tag, \
                    fset, \
                    '{:.1f}'.format(float(overall_acc[iter_tag])*100.0), \
                    '{:.1f}'.format(sum([ppl_accs[fset][name][iter_tag+'_t'] for name in ppl_map])*100.0/len(ppl_map)), \
                    '{:.1f}'.format(np.std([ppl_accs[fset][name][iter_tag+'_t'] for name in ppl_map])*100.0)])

            #if joint:
            #    print('\n\nPeople (' + the_tag + '): ')
            #    for person in ppl_map:
            #        print('\t' + person + ': ' + str(ppl_accs[person][the_tag]))

    print(table.get_string(sort_key=operator.itemgetter(0, 1), sortby="Features"))

    # generate histograms -- 'ls,tv,fq,tc,lv,gv'
    keys = ['', 'ls,tv,fq,tc,lv,gv']#ur,
    our_colors = [colors.pop(colors.index(random.choice(colors))) for i in range(len(keys))]
    assert np.all([keyt in ppl_accs for keyt in keys])
    for tag_type in tag_set:
        if tag_type == 'shared ethnicity':
            continue
        plt.clf()
        color_ind = 0
        print('\nAnalysis for ' + tag_type)
        for keyt in keys:
            # for plotting
            acc_dist = [ppl_accs[keyt][name][tag_type+'_t'] for name in ppl_map]
            #print(acc_dist)
            plt.hist(acc_dist, alpha=0.5, bins=20, label=save2read(keyt), facecolor=our_colors[color_ind])
            color_ind += 1

            # for analysis
            print('\t ' + save2read(keyt))
            print('\t Macro-average over people: ' + str(sum([ppl_accs[keyt][name][tag_type+'_t'] for name in ppl_map])*100.0/len(ppl_map)))
            mavg_o_v = 3 if tag_type == 'relative age' else 2
            tkset = list(tag_set[tag_type])
            if 'unknown' in tkset:
                tkset.remove('unknown')
            print('\t Macro-average over values: ' + str(sum([sum([ppl_accs[keyt][name][tag_type][tag_val+'_rp'][0] for name in ppl_map])*100.0/sum([ppl_accs[keyt][name][tag_type][tag_val+'_rp'][1] for name in ppl_map]) for tag_val in tkset])/mavg_o_v))

            maoptv = sum([np.average([ppl_accs[keyt][name][tag_type][tag_val+'_rp'][0]*100.0/ppl_accs[keyt][name][tag_type][tag_val+'_rp'][1] for name in ppl_map if ppl_accs[keyt][name][tag_type][tag_val+'_rp'][1] > 0]) for tag_val in tkset])/mavg_o_v
            print('\t Macro-average over people then values: ' + str(maoptv))
            #print('\t Macro-averaged: ' + str(sum([ppl_accs[keyt][name][tag_type+'_rp'][0]*100.0/ppl_accs[keyt][name][tag_type+'_rp'][1] for name in ppl_map])/len(ppl_map)))
            print('\t Micro-averaged: ' + str(sum([ppl_accs[keyt][name][tag_type+'_rp'][0] for name in ppl_map])*100.0/sum([ppl_accs[keyt][name][tag_type+'_rp'][1] for name in ppl_map])))
            for tag_val in tag_set[tag_type]:
                #val_dist = [ppl_accs[keyt][name][tag_type][tag_val] for name in ppl_map]
                key_acc = [ppl_accs[keyt][name][tag_type][tag_val] for name in ppl_map if ppl_accs[keyt][name][tag_type][tag_val+'_rp'][1] > 0]
                #print('value distribution: ' + str(val_dist))
                #print('\t\t' + tag_val + ' avg accuracy: ' + str(np.average(val_dist)))
                # this is actually just not counting non-occurs as zero
                print('\t\t' + tag_val + ' avg accuracy: ' + str(np.average(key_acc)))

        plt.title("\n".join(wrap('\nvs '.join([save2read(keyt) for keyt in keys]) + ' for ' + tag_type + ' prediction', 60)))
        plt.legend()#loc='upper right')
        vs_str = '_vs_'.join(['_'.join(fset.split(',')) for  fset in keys])
        if not os.path.exists('log_plots/' + vs_str):
            os.makedirs('log_plots/' + vs_str)
        plt.savefig('log_plots/' + vs_str + '/' + ('_'.join(tag_type.split())) + '_' + vs_str + '.png')

def save2read(name):
    parts = name.split(',')
    outnames = []
    if 'ur' in parts:
        outnames.append('Attributes')
    if 'ls' in parts:
        outnames.append('LIWC')
    if 'tv' in parts:
        outnames.append('Time Values')
    if 'fq' in parts and 'tc' in parts:
        outnames.append('Frequency')
    if 'lv' in parts:
        outnames.append('Style')
    if 'gv' in parts:
        outnames.append('Graph')
    return ', '.join(outnames) if len(outnames) > 0 else 'None'

if __name__ == "__main__":
    main()
