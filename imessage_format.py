

import operator, random, msgpack, re, os
from datetime import datetime
from tqdm import tqdm

import dateutil.parser
import sqlite3

from utils import URL_PATTERN, open_for_write, get_domain, read_filtered, read_people_file, settings

ADDRESS_BOOK_PATH = ''

def main():
    print('Do not run this file. Run "message_formatter.py" to convert messages.')

def read_addressbook(path=ADDRESS_BOOK_PATH):
    abook = {}
    with open(path) as handle:
        lines = handle.readlines()
        for line in lines:
            if line.strip() == '':
                continue
            parts = line.strip().split(':')
            abook[parts[0]] = parts[1]
    return abook

def imsg_format(location, ith, high_sierra=True):
    tcount = 0
    selfcount = 0
    conversations = {}
    f_filter = read_filtered('imsg')
    ppl = read_people_file('imsg_people')
    convos = imsg_get_convos(location, high_sierra)

    print('Number of Conversations: ' + str(len(convos)))
    for tconvo in convos:
        tcount += 1
        print("\n\nLooking at conversation " + str(tcount) + " of " + str(len(convos)) + "...")

        pname = tconvo['id']
        print("Participant: " + ppl[pname])
        mwith = pname
        if ppl[pname] in settings['my_name']:
            print("Conversation with myself. Skipping...")
            selfcount += 1
            continue
        elif mwith in f_filter:
            print("Filtering because name is in filter list...")
            print(f_filter)
            print(mwith)
            continue
        mwith = ppl[mwith]
        print("Adding messages to conversation with " + mwith + "...")
        if mwith in conversations:
            print("Conversation with " + mwith + " already exists...")
        else:
            conversations[mwith] = {}
            conversations[mwith]["with"] = mwith
            conversations[mwith]["messages"] = []

        for event in tconvo['messages']:
            msg_obj = event
            msg_obj['user'] = ppl[msg_obj['user']] if msg_obj['user'] not in settings['my_name'] else settings['my_name'][0]
            if "text" in msg_obj and msg_obj["text"] != "":
                conversations[mwith]["messages"].append(msg_obj)
        print("Conversation size is now: " + str(len(conversations[mwith]["messages"])))

    print('Writing output files...')
    for k,v in conversations.items():
        if len(v['messages']) < 2:
            print('There is only one message with ' + k + ' -- not a conversation. Skipping...')
            continue
        #if k in f_filter:
        #    print("Not a valid conversation with " + k + ". Filtering...")
        #    continue
        sname = k.split()
        handle = open_for_write(settings['DATA_DIR'] + '/IMSG_' + str(ith) + '_' + '_'.join(sname), binary=True)
        handle.write(msgpack.packb(v))
        handle.close()

def gen_db_info(cur):
    rows = []
    for row in cur.execute('SELECT name FROM sqlite_master WHERE type=\'table\''):
        rows.append(row)
    for row in rows:
        print('\nrow is: ' + row[0])
        for ro2 in cur.execute('PRAGMA table_info(' + row[0] + ')'):
            print('\t' + str(ro2))

def switch_mime_type(intype):
    ret = '<ATTACHMENT>'
    if '/' in intype:
        gentype = intype.split('/')[0]
        if gentype == 'image':
            ret = '<PICTURE>'
        elif gentype == 'video':
            ret = '<VIDEO>'
        elif gentype == 'audio':
            ret = '<AUDIO>'
    return ret

def imsg_get_convos(imsg_path, high_sierra=True):
    conn = sqlite3.connect(imsg_path)
    cur = conn.cursor()
    abook = {} if ADDRESS_BOOK_PATH != '' else read_addressbook()

    #gen_db_info(cur)
    #for row in cur.execute('SELECT A.ROWID, B.id, A.text, A.date FROM message AS A JOIN handle AS B on A.handle_id = B.ROWID WHERE handle_id IN (1,2) LIMIT 10'):
    #    print(row)

    #cq = "SELECT DISTINCT chat_id FROM chat_message_join"
    cq = "SELECT DISTINCT ROWID FROM handle"
    cqset = [row[0] for row in cur.execute(cq)]
    #print(cqset)
    print("Number of handle IDs found: " + str(len(cqset)))

    convos = []

    # INNER JOIN chat_message_join T2 ON T2.chat_id=" + str(chat_id) + " AND T1.ROWID=T2.message_id \-- chat_ids include group chats
    for handle_id in cqset:
        convo = {'messages': []}
        print('Querying for handle_id = ' + str(handle_id))
        date_val = 'date/1000000000' if high_sierra else 'date'
        aq = "SELECT T1.ROWID, E.id, is_from_me, text, datetime(" + date_val + " + strftime('%s','2001-01-01'), 'unixepoch') as date_utc, T5.mime_type \
            FROM message T1 \
            JOIN handle E ON T1.handle_id = E.ROWID \
            LEFT OUTER JOIN message_attachment_join T4 ON T4.message_id = T1.ROWID \
            LEFT OUTER JOIN attachment T5 ON T4.attachment_id = T5.ROWID \
            WHERE T1.handle_id = " + str(handle_id) + " AND T1.cache_roomnames IS NULL AND T1.text IS NOT NULL \
            ORDER BY T1.date"# LIMIT 50
        result_set = [row for row in cur.execute(aq)]
        if len(result_set) == 0:
            print('No messages in this conversation...')
            continue
        convo['name'] = result_set[0][1]
        convo['id'] = convo['name']

        # try to look up in address book
        tconvo_id = convo['id'] if not convo['id'].startswith('+1') else convo['id'][2:]
        tconvo_id = tconvo_id if not tconvo_id.startswith('+') else tconvo_id[1:]
        if tconvo_id in abook:
            convo['name'] = abook[tconvo_id]

        for row in result_set:#tqdm?
            formatted_date = dateutil.parser.parse(row[4])
            if str(row[4]) != str(formatted_date):
                print('Date Warning: ' + str(row[4]) + " --> " + str(formatted_date))
            msg_text = row[3] if row[5] == None else row[3].replace(u"\ufffc", ' ' + switch_mime_type(row[5]) + ' ')
            #if row[5] != None:
            #    print(row)
            #    print(msg_text)
            message = {'user': settings['my_name'][0] if row[2] == 1 else row[1], 'text': row[3], 'date': formatted_date.isoformat()}
            convo['messages'].append(message)
        convo['num_events'] = len(convo['messages'])
        convos.append(convo)
    return convos

def imsg_msg_sample(convo, num_msg=15):
    num_events = convo['num_events']
    rc_ind = num_events - num_msg if num_events > num_msg else 0
    rc_ind = round(random.random()*rc_ind)
    msg_sample_list = []

    for e_ind in range(rc_ind, rc_ind+10):
        if e_ind >= num_events:
            break
        msg_obj = convo['messages'][e_ind]
        msg_sample_list.append(str(msg_obj["user"]) + " (" + str(msg_obj["date"]) + "): " + msg_obj["text"])
    return msg_sample_list

if __name__ == "__main__":
    main()
