

import operator, random, msgpack, json, sys, re, os
from datetime import datetime

import dateutil.parser

from utils import URL_PATTERN, open_for_write, get_domain, read_people_file, read_filtered, settings

def main():
    print('Do not run this file. Run "message_formatter.py" to convert messages.')

def test_ig_load():
    ig_path = settings['INSTAGRAM_PATH'][0]
    print(ig_path)
    with open(ig_path) as handle:
        msg_data = json.load(handle)
    keys_all = set()
    for i in range(len(msg_data)):
        parti = msg_data[i]['participants']
        if len(parti) > 2:
            print('Group conversation, skipping...')
            continue
        other_person = parti[0] if parti[0] not in settings['my_name'] else parti[1]
        print('The other person is: ' + str(other_person))
        for msg in msg_data[i]['conversation']:
            msg_text = None
            #if 'likes' in msg or 'story_share' in msg or 'link' in msg:
            #    print(msg)
            if 'text' in msg:
                msg_text = msg['text']
            elif 'media_share_caption' in msg:
                msg_text = msg['media_share_caption']
            if 'media' in msg:
                if msg_text == None:
                    msg_text = '<MEDIA>'
                else:
                    msg_text += ' <MEDIA>'
            if msg_text == None:
                print(msg)
            else:
                msg_obj = {'user': msg['sender'], 'text': msg_text, 'date': dateutil.parser.parse(msg['created_at'])}
                print(msg_obj)
            #for key in msg.keys():
            #    keys_all.add(key)
    #print('\n' + str(keys_all))

def ig_format(location, ith):
    conversations = {}
    tcount = 0
    f_filter = read_filtered('ig')
    aliases = read_people_file('ig_people')
    convos = ig_get_convos(location)

    print('Number of Conversations: ' + str(len(convos)))
    for tconvo in convos:
        tcount += 1
        print('\n\nLooking at conversation ' + str(tcount) + ' of ' + str(len(convos)) + '...')

        pname = tconvo['id']
        print('Participant: ' + aliases[pname])
        mwith = pname
        if aliases[pname] in settings['my_name']:
            msg_sample = ig_msg_sample(tconvo)
            print('Length of sample: ' + str(len(tconvo['messages'])))
            for msg_p in msg_sample:
                print(msg_p)
            print('Error: Conversation with self!')
            sys.exit(0)
        elif mwith in f_filter:
            print('Filtering because name is in filter list...')
            print(f_filter)
            print(mwith)
            continue
        mwith = aliases[mwith] if mwith in aliases else mwith
        print('Adding messages to conversation with ' + mwith + '...')
        if mwith in conversations:
            print('Conversation with ' + mwith + ' already exists...')
        else:
            conversations[mwith] = {}
            conversations[mwith]['with'] = mwith
            conversations[mwith]['messages'] = []
        #print(n1["conversation_state"]["conversation_id"])
        for event in tconvo['messages']:
            msg_obj = event
            msg_obj['user'] = aliases[msg_obj['user']] if msg_obj['user'] in aliases else msg_obj['user']
            if 'text' in msg_obj and msg_obj['text'] != '' and msg_obj['text'] != None:
                conversations[mwith]['messages'].append(msg_obj)
        print('Conversation size is now: ' + str(len(conversations[mwith]['messages'])))

    print('\nSorting conversation messages by date...')
    for k,v in conversations.items():
        v['messages'].sort(key=lambda r: r['date'])

    print('Writing output files...')
    for k,v in conversations.items():
        if len(v['messages']) < 2:
            print('There is only one message with ' + k + ' -- not a conversation. Skipping...')
            continue
        handle = open_for_write(settings['DATA_DIR'] + "/IG_" + str(ith) + "_" + "_".join(k.split()), binary=True)
        if 'messages' in v:
            for msg in v['messages']:
                msg['date'] = msg['date'].isoformat()
        handle.write(msgpack.packb(v))
        handle.close()

def ig_get_convos(ig_path):
    convos = []

    with open(ig_path) as handle:
        msg_data = json.load(handle)
    keys_all = set()
    for i in range(len(msg_data)):
        parti = msg_data[i]['participants']
        print('Participants: ' + str(parti))
        if len(parti) > 2:
            print('Group conversation, skipping...')
            continue
        other_person = parti[0] if parti[0] not in settings['my_name'] else parti[1]
        print('The other person is: ' + str(other_person))

        messages = []
        for msg in msg_data[i]['conversation']:
            msg_text = None
            #if 'likes' in msg or 'story_share' in msg or 'link' in msg:
            #    print(msg)
            if 'text' in msg:
                msg_text = msg['text']
            elif 'media_share_caption' in msg:
                msg_text = msg['media_share_caption']
            if 'media' in msg:
                if msg_text == None:
                    msg_text = '<MEDIA>'
                else:
                    msg_text += ' <MEDIA>'
            if msg_text == None:
                print(msg)
            else:
                msg_obj = {'user': msg['sender'], 'text': msg_text, 'date': dateutil.parser.parse(msg['created_at'])}
                messages.append(msg_obj)

        convos.append({'name': other_person, 'place': i, 'num_events': len(messages), 'messages': messages, 'id': other_person})

    for v in convos:
        v['messages'].sort(key=lambda r: r['date'])
    return convos

def ig_msg_sample(convo, num_msg=15):
    num_events = convo['num_events']
    rc_ind = num_events - num_msg if num_events > num_msg else 0
    rc_ind = round(random.random()*rc_ind)
    msg_sample_list = []

    for e_ind in range(rc_ind, rc_ind+num_msg):
        if e_ind >= num_events:
            break
        msg_sample_list.append(str(convo['messages'][e_ind]['user']) + " (" + str(convo['messages'][e_ind]['date']) + '): ' + convo['messages'][e_ind]['text'])
    return msg_sample_list

if __name__ == '__main__':
    main()
