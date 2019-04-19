

import operator, msgpack, random, json, sys, os, re
from datetime import datetime

from utils import OTR_CONTEXT, OTR_STRS, open_for_write, get_domain, read_filtered, read_people_file, settings

def main():
    print('Do not run this file. Run "message_formatter.py" to convert messages.')

def gh_format(location, ith):
    conversations = {}
    tcount, selfcount = [0]*2
    f_filter = read_filtered('gh')
    ppl = read_people_file('gh_people')
    convos, ppl_map = gh_get_convos(location, from_annotator=True)

    print('Number of Conversations: ' + str(len(convos)))
    for tconvo in convos:
        tcount += 1
        print("\n\nLooking at conversation " + str(tcount) + " of " + str(len(convos)) + "...")

        pname = tconvo['id']
        print("Participant: " + ppl[pname])
        mwith = pname
        if ppl[pname] in settings['my_name']:
            msg_sample = gh_msg_sample(tconvo, ppl_map)
            print("Length of sample: " + str(len(tconvo['messages'])))
            for msg_p in msg_sample:
                print(msg_p)
            print("Conversation with myself. Skipping...")
            selfcount += 1
            if selfcount > 1:
                print("Error: More than one conversation with yourself! What is happening!?")
                sys.exit(0)
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
        #if settings['GOOGLE_FTYPE'] == 'OLD':
        #    print(n1["conversation_state"]["conversation_id"])
        #else:
        #    print(n1["conversation"]["conversation_id"])
        for event in tconvo['messages']:
            msg_obj = get_event(event)
            msg_obj['user'] = ppl[msg_obj['user']]
            if "text" in msg_obj and msg_obj["text"] != "" and msg_obj["text"] != None:
                conversations[mwith]["messages"].append(msg_obj)
        print("Conversation size is now: " + str(len(conversations[mwith]["messages"])))

    # Sort conversations and write them
    print("\nSorting conversation messages by date...")
    for k,v in conversations.items():
        v["messages"].sort(key=lambda r: r["date"])

    print("Writing output files...")
    for k,v in conversations.items():
        if len(v["messages"]) < 2:
            print("There is only one message with " + k + " -- not a conversation. Skipping...")
            continue
        #if k in f_filter:
        #    print("Not a valid conversation with " + k + ". Filtering...")
        #    continue
        sname = k.split()
        handle = open_for_write(settings['DATA_DIR'] + "/GH_" + str(ith) + "_" + "_".join(sname), binary=True)
        if 'messages' in v:
            for msg in v['messages']:
                msg['date'] = msg['date'].isoformat()
        handle.write(msgpack.packb(v))
        handle.close()

def gh_get_convos(location, from_annotator=False):
    jc = None
    print("Parsing messages as a JSON object...")
    with open(location) as handle:
        jc = json.loads(handle.read())
    ppl_map = gh_id_map(jc)

    convos = []
    for i in range(0, len(jc['conversation_state' if settings['GOOGLE_FTYPE'] == 'OLD' else 'conversations'])):
        if settings['GOOGLE_FTYPE'] == 'OLD':
            n1 = jc["conversation_state"][i]
            c_type = n1["conversation_state"]["conversation"]["type"]
        else:
            n1 = jc["conversations"][i]
            c_type = n1["conversation"]["conversation"]["type"]

        #print(n1["conversation_id"])
        #print(n1["response_header"])
        #for key in n1["conversation_state"]:
        #    print(key)
        if c_type == "GROUP":
            print("This is a group conversation. Skipping...")
            continue
        if settings['GOOGLE_FTYPE'] == 'OLD':
            participants = n1["conversation_state"]["conversation"]["participant_data"]
        else:
            participants = n1["conversation"]["conversation"]["participant_data"]

        for parti in participants:
            pname = parti["id"]["chat_id"]
            name = ppl_map[pname]
            if name in settings['my_name'] and from_annotator:
                continue
            if settings['GOOGLE_FTYPE'] == 'OLD':
                num_events = len(n1['conversation_state']['event'])
                convos.append({'name': name, 'num_events': num_events, 'messages': n1['conversation_state']['event'], 'place': i, 'id': pname})
            else:
                num_events = len(n1['events'])#['conversation']
                convos.append({'name': name, 'num_events': num_events, 'messages': n1['events'], 'place': i, 'id': pname})#['conversation']
    return convos, ppl_map

def gh_msg_sample(convo, ppl_map, num_msg=15):
    num_events = convo['num_events']
    rc_ind = num_events - num_msg if num_events > num_msg else 0
    rc_ind = round(random.random()*rc_ind)
    msg_sample_list = []

    for e_ind in range(rc_ind, rc_ind+10):
        if e_ind >= num_events:
            break
        event = convo['messages'][e_ind]
        msg_obj = get_event(event)
        if msg_obj['user'] in ppl_map:
            msg_obj['user'] = ppl_map[event["sender_id"]["chat_id"]]
        msg_sample_list.append(str(msg_obj["user"]) + " (" + str(msg_obj["date"]) + "): " + str(msg_obj["text"]))
    return msg_sample_list

def gh_id_map(jc):
    people = {}
    for i in range(0, len(jc['conversation_state' if settings['GOOGLE_FTYPE'] == 'OLD' else 'conversations'])):
        if settings['GOOGLE_FTYPE'] == 'OLD':
            n1 = jc["conversation_state"][i]
            c_type = n1["conversation_state"]["conversation"]["type"]
        else:
            n1 = jc["conversations"][i]
            c_type = n1["conversation"]["conversation"]["type"]
        if c_type == "GROUP":
            continue
        if settings['GOOGLE_FTYPE'] == 'OLD':
            participants = n1["conversation_state"]["conversation"]["participant_data"]
        else:
            participants = n1["conversation"]["conversation"]["participant_data"]
        for parti in participants:
            name = "None"
            if "fallback_name" in parti:
                name = parti["fallback_name"]
            people[parti["id"]["chat_id"]] = name
    return people

def get_event(event):
    fullmsg = ""
    otr_in_c = False
    msg_obj = {}
    msg_obj["user"] = event["sender_id"]["chat_id"]
    msg_obj["date"] = datetime.fromtimestamp(float(event["timestamp"])/1000000.0)
    msg_obj["text"] = None
    if "chat_message" in event:
        if "segment" not in event["chat_message"]["message_content"]:
            fullmsg = "<PICTURE>"
        else:
            segs = event["chat_message"]["message_content"]["segment"]
            # Append segments of the message, skipping OTR
            for msg in segs:
                if msg["type"] == 'LINE_BREAK':
                    continue
                if msg["text"] in OTR_STRS or (otr_in_c and msg["text"] in OTR_CONTEXT):
                    otr_in_c = True
                    continue
                else:
                    otr_in_c = False
                if msg["type"] == "LINK":
                    #print("Link message debug: " + str(msg))
                    fullmsg += " " if fullmsg != "" else ""
                    linkmsg = None
                    if msg["link_data"]["link_target"].startswith("mailto:"):
                        linkmsg = "<EMAIL>"
                    else:
                        linkmsg = '<LINK@' + get_domain(msg["text"]) + '>' #'<LINK>'
                    fullmsg += linkmsg
                    if linkmsg == None:
                        print("Error: Link type without message")
                        input()
                elif not msg["text"].startswith("?OTR"):
                    fullmsg += " " if fullmsg != "" else ""
                    fullmsg += msg["text"].strip()
        msg_obj["text"] = fullmsg
    else:
        # These are understood
        if "hangout_event" in event and event["hangout_event"]["event_type"] != "START_HANGOUT" and event["hangout_event"]["event_type"] != "END_HANGOUT":
            print("Warning: Unknown event " + str(event))
        elif "hangout_event" not in event:
            print("Warning: Missing event " + str(event))
    return msg_obj

if __name__ == "__main__":
    main()
