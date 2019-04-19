

import operator, msgpack, random, json, nltk, math, os, re
import utils

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words, wordnet
from collections import defaultdict
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

from contractions import contractions

dir_path = os.path.dirname(os.path.realpath(__file__))

#print(type(wordnet.keys()))
manywords = set(words.words())
wnl = WordNetLemmatizer()
norms = {}

print("Reading normalizations file...")
with open(dir_path + "/normalizations") as handle:
    for line in handle.readlines():
        t_line = line.strip().split(":")
        if len(t_line) == 2:
            norms[t_line[0].lower()] = t_line[1].lower()
print("Normalizations loaded...")

def find_instances(strin, num_inst):
    print('Finding ' + str(num_inst) + ' instances of string \'' + strin + '\'...')
    counter = 0
    file_set = os.listdir(utils.DATA_DIR)
    random.shuffle(file_set)
    for filename in file_set:
        #print("File: " + filename)
        #print("="*30 + "Percent of total complete: " + str(counter*100/len(os.listdir(utils.DATA_DIR))))
        counter += 1
        convo = None
        ctype = filename.split("_")[0]
        with open(utils.DATA_DIR + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"]
        for message in convo[b"messages"]:
            msg_text = message[b"text"].decode()
            tokens = [_t.lower() for _t in nltk.word_tokenize(msg_text) if _t.isalpha()]
            t_notin = []
            for token in tokens:
                _t_ = wnl.lemmatize(token)
                if _t_ == strin:
                    num_inst -= 1
                    print('Instance ' + str(num_inst) + ' by ' + cname.decode() + ': ' + msg_text)
                    if num_inst == 0:
                        return

def main():
    print("Reading normalization set and added words...")
    added_words = []
    with open(dir_path + "/added_words") as handle:
        for line in handle.readlines():
            t_line = line.strip()
            added_words.append(t_line)

    missing_tok = defaultdict(lambda: 0)
    counter = 0
    print("Reading conversation files...")
    file_set = os.listdir(utils.DATA_DIR)
    random.shuffle(file_set)
    for filename in file_set:
        print("File: " + filename)
        print("="*30 + "Percent of total complete: " + str(counter*100/len(os.listdir(utils.DATA_DIR))))
        counter += 1
        convo = None
        ctype = filename.split("_")[0]
        with open(utils.DATA_DIR + "/" + filename, "rb") as handle:
            convo = msgpack.unpackb(handle.read())
        cname = convo[b"with"]
        mcou = 0
        for message in tqdm(convo[b"messages"]):
            #print("="*30 + "Percent of " + filename + " complete: " + str(mcou*100/len(convo[b"messages"])))
            mcou += 1
            msg_text = message[b"text"].decode()
            #msg_text = expand_text(contractions, msg_text)
            #print("after contractions: " + msg_text)
            #msg_text = expand_text(norms, msg_text)
            #print("after normalizations: " + msg_text)
            tokens = [_t.lower() for _t in nltk.word_tokenize(msg_text) if _t.isalpha()]
            t_notin = []
            for token in tokens:
                _t_ = wnl.lemmatize(token)
                if _t_ not in manywords and _t_ not in added_words and token not in manywords and token not in added_words:
                    #t_notin.append(token)
                    missing_tok[token] += 1

        #if counter > 3:
        #break

    out_file = open('unknown_output', 'w')
    for k,v in sorted(missing_tok.items(), key=operator.itemgetter(1), reverse=True):
        #print("[" + "{:,}".format(v) + "] \"" + k + "\"")
        out_file.write("[" + "{:,}".format(v) + "] \"" + k + "\"\n")
    out_file.close()



#                print("*"*50)
#                print("Message text: " + message[b"text"].decode())
#                print("Expanded text: " + msg_text)
#                for _t in t_notin:
#                    print("Missing Token: " + _t)
#                while True:
#                    mapto = input("Enter mapping: ")
#                    mapto = mapto.strip().split(":")
#                    if len(mapto) == 2:
#                        write_mapping(mapto[0], mapto[1])
#                        norms[mapto[0].lower()] = mapto[1].lower()
#                    elif len(mapto) == 1 and mapto[0] != "":
#                        write_words(mapto[0])
#                        added_words.append(mapto[0])
#                    else:
#                        break

def expand_text(expand_set, message):
    tmsg = message.lower()
    used_set = []
    for _tok in expand_set.keys():
        if _tok not in used_set:
            tmsg = re.sub("([^A-Za-z]|^)" + _tok + "([^A-Za-z]|$)", "\\1" + expand_set[_tok].split(" / ")[0] + "\\2", tmsg, flags=re.MULTILINE)
            used_set.append(_tok)
    return tmsg

def expand_text_old(expand_set, message):
    tmsg = message.lower()
    #print(expand_set)
    tokens = [_t.lower() for _t in nltk.word_tokenize(tmsg) if _t.isalpha()]
    used_set = []
    for _tok in tokens:
        if _tok in expand_set.keys() and _tok not in used_set:
            tmsg = tmsg.replace(_tok, expand_set[_tok].split(" / ")[0])
            used_set.append(_tok)
            #print("replacing " + _tok + " with " + expand_set[_tok].split(" / ")[0])
    return tmsg

def write_mapping(str1, str2):
    with open("normalizations", "a") as handle:
        handle.write(str1 + ":" + str2 + "\n")

def write_words(str1):
    with open("added_words", "a") as handle:
        handle.write(str1 + "\n")

if __name__ == "__main__":
    #main()
    find_instances('heh', 500)
