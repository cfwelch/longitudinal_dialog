

import operator, random, msgpack, json, sys, re, os
from datetime import datetime
from bs4 import BeautifulSoup

import dateutil.parser

from utils import URL_PATTERN, open_for_write, get_domain, read_people_file, read_filtered, settings

def main():
    print('Do not run this file. Run "message_formatter.py" to convert messages.')

def fb_format(location, ith):
    conversations = {}
    tcount, selfcount = [0]*2
    f_filter = read_filtered('fb')
    aliases = read_people_file('fb_people')
    # If it's the HTM file read it, otherwise read all files in directory
    convos = fb_get_convos(location) if settings['FACEBOOK_FTYPE'] != 'NEW' else new_fb_get_convos(location)

    print('Number of Conversations: ' + str(len(convos)))
    for tconvo in convos:
        tcount += 1
        print("\n\nLooking at conversation " + str(tcount) + " of " + str(len(convos)) + "...")

        pname = tconvo['id']
        print("Participant: " + aliases[pname])
        mwith = pname
        if aliases[pname] in settings['my_name']:
            msg_sample = fb_msg_sample(tconvo)
            print("Length of sample: " + str(len(tconvo['messages'])))
            for msg_p in msg_sample:
                print(msg_p)
            print("Conversation with myself. Skipping...")
            selfcount += 1
            if selfcount > 1:
                print("Error: More than one conversation with yourself! What is happening!?")
                # sys.exit(0)
            continue
        elif mwith in f_filter:
            print("Filtering because name is in filter list...")
            print(f_filter)
            print(mwith)
            continue
        mwith = aliases[mwith] if mwith in aliases else mwith
        print("Adding messages to conversation with " + mwith + "...")
        if mwith in conversations:
            print("Conversation with " + mwith + " already exists...")
        else:
            conversations[mwith] = {}
            conversations[mwith]["with"] = mwith
            conversations[mwith]["messages"] = []
        #print(n1["conversation_state"]["conversation_id"])
        for event in tconvo['messages']:
            msg_obj = event
            msg_obj['user'] = aliases[msg_obj['user']] if msg_obj['user'] in aliases else msg_obj['user']
            if "text" in msg_obj and msg_obj["text"] != "" and msg_obj["text"] != None:
                conversations[mwith]["messages"].append(msg_obj)
        print("Conversation size is now: " + str(len(conversations[mwith]["messages"])))

    # Feb 2018 -- Facebook format does not include seconds -- do not sort
    #print("\nSorting conversation messages by date...")
    #for k,v in thr_obj.items():
    #    v["messages"].sort(key=lambda r: r["date"])

    print("Writing output files...")
    for k,v in conversations.items():
        if len(v["messages"]) < 2:
            print("There is only one message with " + k + " -- not a conversation. Skipping...")
            continue
        # Remove middle initials...
        sname = k.split()
        while len(sname) > 2:
            if len(sname[1]) == 2 and sname[1][1] == ".":
                del sname[1]
            else:
                break
        handle = open_for_write(settings['DATA_DIR'] + "/FB_" + str(ith) + "_" + "_".join(sname), binary=True)
        if 'messages' in v:
            for msg in v['messages']:
                msg['date'] = msg['date'].isoformat()
        handle.write(msgpack.packb(v))
        handle.close()

def new_fb_get_convos(fb_path):
    #group_list = []
    convos = []
    tcount = 0
    for _d in os.listdir(fb_path):
        tcount += 1
        print(_d)
        if os.path.isfile(fb_path + _d + '/message.json'):
            with open(fb_path + _d + '/message.json') as handle:
                ftext = json.loads(handle.read())
                #print(ftext.keys())
                if 'title' not in ftext:
                    print('Conversation with no title, skipping...')
                    #input()
                    continue
                title = ftext['title']
                print('Title: ' + title)
                if title == 'Facebook User':
                    print('Skipping unknown Facebook user...')
                    continue
                elif 'participants' not in ftext:
                    print('Inactive chat (no participants), skipping...')
                    continue
                participants = ftext['participants']
                print('Participants: ' + str(participants))
                thread_type = ftext['thread_type']
                print(thread_type)
                messages = ftext['messages']
                print(len(messages))
                if len(participants) > 2:
                    #group_list.append(title)
                    print('Skipping group conversation...')
                    continue

                thr_obj = {}
                thr_obj["with"] = title
                thr_obj["messages"] = []
                no_sender = False
                for msg in messages:
                    msg_obj = {}
                    if 'content' not in msg:
                        print(msg)
                    #msg_obj['text'] = msg['content']
                    if 'content' in msg:
                        msg_obj['text'] = re.sub(URL_PATTERN, lambda x: '<LINK@' + get_domain(x.group()) + '>', msg['content'])
                    else:
                        msg_obj['text'] = '<STICKER>' if 'sticker' in msg else ''
                    msg_obj['date'] = datetime.fromtimestamp(msg['timestamp_ms']/1000)
                    if 'sender_name' not in msg:
                        no_sender = True
                        break
                    msg_obj['user'] = msg['sender_name']
                    thr_obj["messages"].append(msg_obj)

                if no_sender:
                    print('Skipping conversation with no sender...')
                    continue

                convos.append({'name': title, 'num_events': len(messages), 'messages': thr_obj['messages'], 'place': tcount, 'id': title})

    for v in convos:
        v["messages"].sort(key=lambda r: r["date"])
    return convos

def fb_get_convos(fb_path):
    # If it's the HTM file read it, otherwise read all files in directory
    lines = []
    if os.path.isdir(fb_path):
        for filename in os.listdir(fb_path):
            if filename.endswith("html"):
                #print("File: " + filename)
                with open(fb_path + filename) as handle:
                    lines.extend(handle.readlines())
    else:
        with open(fb_path) as handle:
            lines = handle.readlines()
    lines = "\n".join(lines)

    # Parse the HTML
    print("Parsing HTML...")
    parsed_html = BeautifulSoup(lines, "html.parser")
    threads = parsed_html.findAll("div", {"class": "thread"})
    print("Found " + str(len(threads)) + " threads!")

    convos = []
    tcount = 0
    for thread in threads:
        tcount += 1
        # Get the name of who you're talking to
        try:
            thtemp = str(thread)[str(thread).index("Participants"):]
        except:
            print(str(thread))
            #print("Error: Participants tag not found in HTML")
            print("Participants tag not found in HTML. Skipping...")
            #sys.exit(0)
            continue
        thtemp = thtemp[14:]
        mppl = thtemp[:thtemp.index("<")]
        mppl = mppl.split(",")
        mppl = [mp_t.strip() for mp_t in mppl]
        #print("People: " + str(mppl))
        # If no people in conversation or only you
        if len(mppl) == 0 or (len(mppl) == 1 and mppl[0] in settings['my_name']):
            #print("Conversation with no one?")#input()
            continue
        if len(mppl) == 1 and mppl[0] in settings['my_name']:
            print("Conversation with " + settings['my_name'] + ", skipping...")
            continue
        elif len(mppl) > 2:
            print("This is a group conversation. Skipping...")
            continue
        mwith = mppl[0] if mppl[0] not in settings['my_name'] else mppl[1]
        thr_obj = {}
        thr_obj["with"] = mwith
        thr_obj["messages"] = []
        # Skip if more than one person
        if len(mppl) > 1:
            #print("This is a group conversation. Skipping...")
            continue
        messages = thread.findAll("div", {"class": "message"})
        msg_obj = None
        for child in thread.findChildren():
            # there don't appear to ever be more than one class on an element here
            t_class = child.get("class")[0] if child.get("class") != None else None
            #print(str(child) + " -------------------------- " + str(t_class))
            if t_class == "message_header":
                if msg_obj != None:
                    thr_obj["messages"].insert(0, msg_obj)
                msg_obj = {}
            elif t_class == "user":
                msg_obj["user"] = child.text
            elif t_class == "meta":
                #msg_obj["date"] = datetime.strptime(child.text, "%A, %B %d, %Y at %H:%M%p %Z")
                try:
                    msg_obj["date"] = dateutil.parser.parse(child.text)
                except:
                    print("Meta tag that isn't date... skipping...")
            elif child.name == "p":
                #msg_obj["text"] = child.text
                msg_obj["text"] = re.sub(URL_PATTERN, lambda x: '<LINK@' + get_domain(x.group()) + '>', child.text)
                #if mb4 != msg_obj["text"]:
                    #print(mb4 + " --> " + msg_obj["text"])
                #print(msg_obj)
                if msg_obj["text"].strip() == "":# These seem to be stickers (December 2017)
                    msg_obj["text"] = "<STICKER>"
                    #input()
        # add the last object
        if msg_obj != None:
            thr_obj["messages"].insert(0, msg_obj)
        new_name = mwith

        num_events = len(thr_obj["messages"])
        convos.append({'name': new_name, 'num_events': num_events, 'messages': thr_obj['messages'], 'place': tcount, 'id': new_name})
    return convos

def fb_msg_sample(convo, num_msg=15):
    num_events = convo['num_events']
    rc_ind = num_events - num_msg if num_events > num_msg else 0
    rc_ind = round(random.random()*rc_ind)
    msg_sample_list = []

    for e_ind in range(rc_ind, rc_ind+10):
        if e_ind >= num_events:
            break
        msg_sample_list.append(str(convo["messages"][e_ind]["user"]) + " (" + str(convo["messages"][e_ind]["date"]) + "): " + convo["messages"][e_ind]["text"])
    return msg_sample_list

if __name__ == "__main__":
    main()
