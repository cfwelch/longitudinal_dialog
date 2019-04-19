

import subprocess, operator, random, msgpack, nltk, math, sys, os
from prettytable import PrettyTable
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from collections import Counter

import numpy as np
import dateutil.parser

from utils import START_TIME, settings, liwc_keys, open_for_write, read_people_file, tag_set
WIDTH_ITERATIONS = 3

def make_mention_graph(cutoff, num_ppl):
    PTU, first_names = read_mention_names(num_ppl)

    valid_people = read_people_file("tagged_people")
    ptypes = set()
    # get people types
    for name in valid_people:
        ptypes.add(to_group_string(valid_people[name]))

    for ptype in ptypes:
        print(ptype)
    print("Number of types: " + str(len(ptypes)))

    print("first names: " + str(first_names))
    #init to set of list for negative edges
    mentions = {name: {n2: 0 for n2 in PTU} for name in PTU}
    #mentions = {ptype: {pt2: 0 for pt2 in ptypes} for ptype in ptypes}

    print("Reading conversation files...")
    for filename in os.listdir(settings['DATA_DIR']):
        print("File: " + filename)
        convo = None
        with open(settings['DATA_DIR'] + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"].decode()
        if cname not in PTU:
            print(cname + " is not in the list of top people...")
            continue

        for message in tqdm(convo[b"messages"]):
            if str(message[b"date"]) < str(START_TIME):
                continue
            if b"text" not in message:
                continue
            msg_text = message[b"text"]
            if type(msg_text) == bytes:
                msg_text = msg_text.decode()
            msg_text = msg_text.lower()

            mdate = dateutil.parser.parse(message[b"date"])

            # add up tokens and number of incoming and outgoing messages
            tokens = [_t for _t in nltk.word_tokenize(msg_text)]

            for token in tokens:
                if token in first_names:
                    #g_cname = to_group_string(valid_people[cname])
                    #g_fname = to_group_string(valid_people[first_names[token]])
                    #if g_fname in mentions[g_cname]:
                    #    mentions[g_cname].remove(g_fname)
                    #if first_names[token] in mentions[cname]:
                    #    mentions[cname].remove(first_names[token])

                    # count for edge width
                    if first_names[token] != cname:
                        mentions[cname][first_names[token]] += 1
                    #g_cname = to_group_string(valid_people[cname])
                    #g_fname = to_group_string(valid_people[first_names[token]])
                    #if g_fname != g_cname:
                    #    mentions[g_cname][g_fname] += 1

            if message[b"user"].decode() in settings['my_name']:# if the person is me
                pass
            else:# count tokens from other person
                pass

    make_dotty(mentions, cutoff)

def to_group_string(person):
    # only not using 'shared ethnicity'
    gstr = "male" if person["same gender"] == "yes" else "female"
    gstr += "_"
    gstr += "family_" if person["family"] == "yes" else ""
    #gstr += "school_" if person["school"] == "from school" else ""
    gstr += "work_" if person["work"] == "yes" else ""
    #gstr += "girlfriend_" if person["non-platonic relationship"] == "yes" else ""
    #gstr += "USA" if person["same childhood country"] == "yes" else "non-USA"
    #gstr += person["relative age"]
    return gstr

def make_dotty(ppl_map, cutoff):
    # delete people with mentions in/out not greater than cutoff
    kdels = []
    for key in ppl_map:
        will_del = True
        for key2 in ppl_map:
            if ppl_map[key][key2] > cutoff or ppl_map[key2][key] > cutoff:
                tweight = ppl_map[key2][key] if ppl_map[key2][key] > ppl_map[key][key2] else ppl_map[key][key2]
                print(key + ' has an edge with ' + key2 + ' with weight ' + str(tweight))
                will_del = False
                break
        if will_del:
            kdels.append(key)
    print('Nodes to remove: ' + str(kdels))
    for key in kdels:
        del ppl_map[key]

    maxc, minc = [1, 999999]
    # get max/min
    for name in ppl_map:
        for n2 in ppl_map[name]:
            if ppl_map[name][n2] > maxc:
                maxc = ppl_map[name][n2]
            if ppl_map[name][n2] < minc and ppl_map[name][n2] > cutoff:
                minc = ppl_map[name][n2]

    # try a few times to make a wide image and not a tall one
    widest, wbest = [0, 0]
    for nth_try in range(WIDTH_ITERATIONS):
        # write dot file
        with open('stats/' + settings['prefix'] + '/mgraph' + str(nth_try) + '.dot', 'w') as handle:
            handle.write('#cutoff\t' + str(cutoff) + '\n#nppl\t' + str(len(ppl_map)) + '\n')
            handle.write('digraph G {\n\tgraph [pad="0.1", nodesep="0.1", ranksep="0.3"];\n\n\tsubgraph {\n')
            node_name = {}
            node_lines = []
            for name in ppl_map:
                node_name[name] = 'n' + str(len(node_name))
                node_lines.append('\t\t' + node_name[name] + '[label="' + name.replace("_", "\\n") + '"];\n')#str(len(node_name))
            random.shuffle(node_lines)
            for nline in node_lines:
                handle.write(nline)
            for name in ppl_map:
                for m in ppl_map[name]:
                    if m != name and ppl_map[name][m] > cutoff:
                        handle.write('\t\t' + node_name[name] + ' -> ' + node_name[m] + ' [penwidth=' + str((ppl_map[name][m]-minc)*5.0/(maxc-minc)+0.5) + ']\n')
            handle.write('\t}\n}\n')

        proc = subprocess.Popen(['dot', '-Tpng', 'mgraph' + str(nth_try) + '.dot', '-o', 'mgraph' + str(nth_try) + '.png'], cwd=r'./stats/' + settings['prefix'])
        proc.communicate()

        im = Image.open('stats/' + settings['prefix'] + '/mgraph' + str(nth_try) + '.png')
        width, height = im.size
        if width > widest:
            widest = width
            wbest = nth_try

    for nth_try in range(WIDTH_ITERATIONS):
        if nth_try != wbest:
            os.remove('stats/' + settings['prefix'] + '/mgraph' + str(nth_try) + '.png')
            os.remove('stats/' + settings['prefix'] + '/mgraph' + str(nth_try) + '.dot')
    os.rename('stats/' + settings['prefix'] + '/mgraph' + str(wbest) + '.png', 'stats/' + settings['prefix'] + '/mgraph.png')
    os.rename('stats/' + settings['prefix'] + '/mgraph' + str(wbest) + '.dot', 'stats/' + settings['prefix'] + '/mgraph.dot')

# Set num_ppl to a number greater than zero to cut off the number of people in the generated graph.
def read_mention_names(num_ppl=-1):
    # read nicknames file
    nicknames = []
    with open('nicknames', 'r') as handle:
        for line in handle.readlines():
            lp = line.strip()
            if lp.startswith('#') or lp == '':
                continue
            lp = lp.split(':')
            if len(lp) != 2:
                print('Error reading this line -- skipping: ' + str(lp))
            else:
                nicknames.append([lp[0].strip(), lp[1].strip()])

    # read valid people
    P_BY_M = []
    total_msg_stats = None
    #with open('stats/' + settings['prefix'] + '/total_messages_per_person.csv') as handle:
    with open('stats/' + settings['prefix'] + '/people_msg_order_desc') as handle:
        total_msg_stats = handle.readlines()
        for line in total_msg_stats:#[1:]:
            P_BY_M.append(line.split('\t')[0])

    if num_ppl > 0:
        P_USE = num_ppl if num_ppl < len(P_BY_M) else len(P_BY_M)
        PTU = [P_BY_M[i] for i in range(0, P_USE)]
    else:
        PTU = P_BY_M

    first_names = {name.split()[0].lower(): name for name in PTU}
    print('Loaded ' + str(len(first_names)) + ' first names...')
    if len(first_names) != len(PTU):
        print('Warning: First name collision ' + str(len(first_names)) + ' < ' + str(len(PTU)) + '.')
        #print(Counter([name.split()[0].lower() for name in PTU]))
        fn_coll = [fnk for fnk, fnv in dict(Counter([name.split()[0].lower() for name in PTU])).items() if fnv > 1]
        print('Collision elements: ' + str(fn_coll))
    print('Loaded ' + str(len(nicknames)) + ' nicknames...')
    for name in nicknames:
        if name[1] in PTU:
            first_names[name[0]] = name[1]

    return PTU, first_names

if __name__ == "__main__":
    make_mention_graph(25, 20)
