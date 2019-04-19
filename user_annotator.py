

import operator, msgpack, random, time, json, os, sys

from datetime import datetime
from pprint import pprint
from bs4 import BeautifulSoup
from prettytable import PrettyTable
from argparse import ArgumentParser

import dateutil.parser

from utils import MSG_TYPE_PATHS, MSG_TYPE, MIN_MESSAGES, open_for_write, read_people_file, tag_set, tag_description, write_filtered, read_filtered, write_people, settings
from google_format import gh_msg_sample, gh_get_convos
from facebook_format import fb_msg_sample, fb_get_convos, new_fb_get_convos
from imessage_format import imsg_msg_sample, imsg_get_convos
from instagram_format import ig_msg_sample, ig_get_convos

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
	# Parse arguments
	parser = ArgumentParser()
	parser.add_argument("-sp1", "--skip-phase1", dest="skip_phase1", help="Skip phase 1: label and merge", default=False, action="store_true")
	parser.add_argument("-sp2", "--skip-phase2", dest="skip_phase2", help="Skip phase 2: add tags", default=False, action="store_true")
	parser.add_argument("-auto", "--auto", dest="auto", help="For testing purposes this auto random tags to all people", default=False, action="store_true")
	cmdopts = parser.parse_args()

	# Begin messages
	print(("="*50) + "\n\t\tUser Annotator\n" + ("="*50))
	print("This script will allow you to annotate and merge contacts from your personal chat corpus.")
	print("You will be shown users and asked a set of questions about each. Your options will be presented at each step.")

	# Get people
	if not cmdopts.skip_phase1:
		print(("="*50) + "\n\t\tPhase 1\n" + ("="*50))
		print("There are likely some messages from automated systems. These should be filtered.")
		print("Other actions will include merging contacts, renaming people, and annotating attributes of those people.")
		print("Press enter to begin...")
		input()
		for location in MSG_TYPE_PATHS['GH']:
			get_people('GH', location, cmdopts.auto)
		for location in MSG_TYPE_PATHS['FB']:
			get_people('FB', location, cmdopts.auto)
		for location in MSG_TYPE_PATHS['IMSG']:
			get_people('IMSG', location, cmdopts.auto)
		for location in MSG_TYPE_PATHS['IG']:
			get_people('IG', location, cmdopts.auto)
		# Do the SMS part automatically -- kind of hacky
		# auto_filter('sms')
		# auto_filter('aim')
	else:
		print("Skipping phase 1: label and merge...")

	people_counts, full_ppl = [None]*2
	# Add tags
	if not cmdopts.skip_phase2:
		people_tags = read_people_file("tagged_people")
		print("Figuring out who should be tagged...")
		people_counts = get_msg_counts()
		print("Message counts for people: " + str(people_counts))
		print(("="*50) + "\n\t\tPhase 2\n" + ("="*50))
		print("This step will read people written to files and ask you to add tags to them.")
		print("The tag type will be shown along with your options.\n")
		print("The tags are defined as follows:")
		for tag in tag_description:
			print("\t" + tag + ": " + tag_description[tag])
		print("\nPress enter to begin...")
		input()
		full_ppl = get_full_ppl()

		# Tag everyone
		done_tagged = 0
		print("Tagging people with more than " + str(MIN_MESSAGES) + " messages!")
		for k,v in full_ppl.items():
			done_tagged += 1
			if k in settings['my_name'] or people_counts[k] < MIN_MESSAGES:
				continue
			print("\n" + "="*50 + "\nTagging person " + str(done_tagged) + " of " + str(len(full_ppl)))
			print("Name: " + ", ".join(v) + " -> " + k + "\n" + ("="*50))
			if k not in people_tags:
				people_tags[k] = {}
			for tag in tag_set.keys():
				if tag in people_tags[k]:
					print("Already tagged " + k + " for " + tag + ", skipping...")
					continue
				print("Tag name: " + tag)
				if not cmdopts.auto:
					options_str = "Options: " + ", ".join(["(" + opt[0].upper() + ")" + opt[1:] for opt in tag_set[tag]])
					opt_lookup = {opt[0].lower(): opt for opt in tag_set[tag]}
					print(options_str)
					done = False
					while not done:
						opt = input().lower().strip()
						opt = opt[0] if len(opt) > 0 else " "
						if opt in opt_lookup:
							print("Labeling " + tag + " as " + opt_lookup[opt] + "...")
							people_tags[k][tag] = opt_lookup[opt]
							done = True
						else:
							print(options_str)
				else:
					rand_tag = random.choice(tag_set[tag])
					print("Randomly assigning label '" + rand_tag + "'...")
					people_tags[k][tag] = rand_tag
				print("Writing " + k + " to file...")
				write_people("tagged", people_tags)
	else:
		print("Skipping phase 2: add tags...")

	# Show stats
	print("\n" + "="*50 + "\n\tShowing Statistics for People Tagged" + "\n" + "="*50)
	tagged_people = read_people_file("tagged_people")
	agg_stats = {tkey: [0]*len(tag_set[tkey]) for tkey in tag_set}
	agg_msg = {tkey: [0]*len(tag_set[tkey]) for tkey in tag_set}
	if people_counts == None:
		people_counts = get_msg_counts()
	print(people_counts)
	tagged_counts = {}
	print("\n\nPeople in people count map with more than " + str(MIN_MESSAGES) + " but not in full people map:")
	if full_ppl == None:
		full_ppl = get_full_ppl()
	for perso_ in people_counts:
		if people_counts[perso_] > MIN_MESSAGES and perso_ not in full_ppl:
			print("\t" + perso_ + ": " + str(people_counts[perso_]))
		elif people_counts[perso_] > MIN_MESSAGES and perso_ in full_ppl:
			tagged_counts[perso_] = people_counts[perso_]

	for person in tagged_people:
		if person not in people_counts:
			people_counts[person] = 0
	for person in sorted(tagged_people):
		#print(person + ": " + str(tagged_people[person]))
		#print("Messages: " + str(sum([people_counts[t_name] for t_name in full_ppl[person]])) + "\n")
		for tag in tag_set:
			agg_stats[tag][tag_set[tag].index(tagged_people[person][tag])] += 1
			agg_msg[tag][tag_set[tag].index(tagged_people[person][tag])] += people_counts[person]

	handle = open_for_write('stats/' + settings['prefix'] + '/people_msg_order_desc')
	for k,v in sorted(tagged_counts.items(), key=operator.itemgetter(1), reverse=True):
		if k == '':
			continue
		handle.write(k + '\t' + str(v) + '\n')
	handle.close()

	# sums should be the same
	temp = [sum([agg_msg[tag][v] for v in range(0, len(tag_set[tag]))]) for tag in tag_set]
	total_messages = temp[0]
	for v in range(1, len(temp)):
		assert temp[v] == temp[0]

	print("\nPeople Statistics: ")
	header_set = "Group\tValue\tPeople\tMessages\t%People\t%Messages"
	table = PrettyTable(header_set.split("\t"))
	table.align["Group"] = "l"
	handle = open_for_write("stats/" + settings['prefix'] + "/total_group_stats.csv")
	handle.write(header_set + "\n")
	for tag in agg_stats:
		for v in range(0, len(tag_set[tag])):
			table_row = tag + "\t" + tag_set[tag][v] + "\t" + str(agg_stats[tag][v]) + "\t" + str(agg_msg[tag][v]) + "\t" + str(agg_stats[tag][v]*100.0/len(tagged_people)) + "\t" + str(agg_msg[tag][v]*100.0/total_messages)
			handle.write(table_row + "\n")
			table.add_row(table_row.split("\t"))
	handle.close()
	print(table.get_string(sort_key=operator.itemgetter(0), sortby="Group"))

def get_full_ppl():
	ppl_types = {mtype.lower(): {'people': [], 'filtered': None} for mtype in MSG_TYPE}
	for mtype in MSG_TYPE:
		ppl_types[mtype.lower()]['people'] = read_people_file(mtype.lower() + '_people')
		ppl_types[mtype.lower()]['filtered'] = read_filtered(mtype.lower())

	full_ppl = {}
	for ppl_type_i in ppl_types:
		for k,v in ppl_types[ppl_type_i]['people'].items():
			if k in ppl_types[ppl_type_i]['filtered']:
				continue
			if v not in full_ppl:
				full_ppl[v] = []
			full_ppl[v].append(k)
	return full_ppl

def get_msg_counts():
	people = {}

	# SMS
	sms_ppl = read_people_file("sms_people")
	files = os.listdir(settings['DATA_DIR'])
	for file in files:
		if file.startswith("SMS_"):
			tname = " ".join(file.split("_")[1:])
			convo = None
			with open(settings['DATA_DIR'] + "/" + file, "rb") as handle:
				convo = msgpack.unpackb(handle.read())
			cname = convo[b"with"].decode()
			tname = sms_ppl[tname]
			if tname not in people:
				people[tname] = 0
			people[tname] += len(convo[b"messages"])

	# AIM
	aim_ppl = read_people_file("aim_people")
	files = os.listdir(settings['DATA_DIR'])
	for file in files:
		if file.startswith("AIM_"):
			tname = " ".join(file.split("_")[2:])
			convo = None
			with open(settings['DATA_DIR'] + "/" + file, "rb") as handle:
				convo = msgpack.unpackb(handle.read())
			cname = convo[b"with"].decode()
			tname = aim_ppl[tname]
			if tname not in people:
				people[tname] = 0
			people[tname] += len(convo[b"messages"])

	# FB
	fb_ppl = read_people_file('fb_people')
	for location in MSG_TYPE_PATHS['FB']:
		convos = fb_get_convos(location) if settings['FACEBOOK_FTYPE'] != 'NEW' else new_fb_get_convos(location)
		for convo in convos:
			mwith = fb_ppl[convo['id']]
			if mwith not in people:
				people[mwith] = 0
			people[mwith] += len(convo['messages'])

	# IG
	ig_ppl = read_people_file('ig_people')
	for location in MSG_TYPE_PATHS['IG']:
		convos = ig_get_convos(location)
		for convo in convos:
			mwith = ig_ppl[convo['id']]
			if mwith not in people:
				people[mwith] = 0
			people[mwith] += len(convo['messages'])

	# GH
	gh_ppl = read_people_file('gh_people')
	for location in MSG_TYPE_PATHS['GH']:
		convos, ppl_map = gh_get_convos(location, from_annotator=True)
		for convo in convos:
			name = gh_ppl[convo['id']]
			if name not in people:
				people[name] = 0
			people[name] += len(convo['messages'])

	# IMSG
	imsg_ppl = read_people_file('imsg_people')
	for location in MSG_TYPE_PATHS['IMSG']:
		convos = imsg_get_convos(location)
		for convo in convos:
			name = imsg_ppl[convo['id']]
			if name not in people:
				people[name] = 0
			people[name] += len(convo['messages'])

	return people

def auto_filter(prefix):
	filtered_people = read_filtered(prefix)
	okay_people = []
	for imsgtype in ['fb', 'gh', 'imsg']:
		for k,v in read_people_file(imsgtype + '_people').items():
			if k not in filtered_people:
				okay_people.append(v)

	files = os.listdir(settings['DATA_DIR'])
	people = {}
	for file in files:
		if file.startswith(prefix.upper() + '_'):
			# I don't know if this was different at some point on a different machine but I think it should always be split("_")[1:] for the auto-files (SMS/AIM)
			# ^ I thought this previous statement was true but I went back to look at the AIM formatter and it does add the number prefix in -- not sure why I had different AIM parsed files before
			tname = " ".join(file.split("_")[1:]) if prefix != 'aim' else " ".join(file.split("_")[2:])
			# tname = ' '.join(file.split('_')[1:])
			convo = None
			with open(settings['DATA_DIR'] + "/" + file, "rb") as handle:
				convo = msgpack.unpackb(handle.read())
			cname = convo[b"with"].decode()
			if len(convo[b"messages"]) < 3 and cname not in okay_people:# was MIN_MESSAGES
				filtered_people.append(tname)
				print("Adding " + tname + " to filtered people because of " + prefix.upper() + " count...")
			people[tname] = tname
	write_filtered(filtered_people, prefix)
	write_people(prefix, people)

def get_people(msg_prefix, location, auto):
	print('\n\nReading messages of type ' + msg_prefix + ' from ' + location + '...')
	msg_prefix = msg_prefix.lower()
	people = read_people_file(msg_prefix + '_people')
	filtered_people = read_filtered(msg_prefix)

	convos, ppl_map = [None]*2
	if msg_prefix == 'fb':
		convos = fb_get_convos(location) if settings['FACEBOOK_FTYPE'] != 'NEW' else new_fb_get_convos(location)
	elif msg_prefix == 'gh':
		convos, ppl_map = gh_get_convos(location, from_annotator=True)
		# add you to the people
		mk = None
		mv = None
		for ik in ppl_map:
			if ppl_map[ik] in settings['my_name']:
				mk = ik
				mv = ppl_map[ik]
		assert mk != None
		if mv not in people:
			people[mk] = mv
	elif msg_prefix == 'imsg':
		convos = imsg_get_convos(location)
	elif msg_prefix == 'ig':
		convos = ig_get_convos(location)

	print("Number of conversations: " + str(len(convos)))
	for i in range(0, len(convos)):
		print("Looking at conversation " + str(i+1) + " of " + str(len(convos)) + "...")

		if convos[i]['id'] in people:
			print('Already tagged ' + str(convos[i]['id']) + ', continuing...')
			continue
		if convos[i]['num_events'] < 3:
			filtered_people.append(convos[i]['id'])
			write_filtered(filtered_people, msg_prefix)
			#people[thread['name']] = thread['name']
			people[convos[i]['id']] = convos[i]['name']
			write_people(msg_prefix, people)
			continue

		print("\n" + "-"*50)
		print("Found person with chat ID " + str(convos[i]['id']) + " and name " + str(convos[i]['name']) + ". Is this persons name correct?")

		if not auto:
			options_str = "Options: (Y)es, (N)o, (F)ilter, (V)iew conversation"
			print(options_str)
			done = False
			while not done:
				opt = input().lower().strip()
				opt = opt[0] if len(opt) > 0 else " "
				if opt == "n":
					print("What is this person's name?")
					new_name = input()
					convos[i]['name'] = new_name
					done = True
					print("Adding " + str(convos[i]['name']) + " to people...")
				elif opt == "v":
					print("Total number of messages: " + str(convos[i]['num_events']))
					msg_sample_list = []
					if msg_prefix == 'gh':
						msg_sample_list = gh_msg_sample(convos[i], ppl_map)
					elif msg_prefix == 'fb':
						msg_sample_list = fb_msg_sample(convos[i])
					elif msg_prefix == 'ig':
						msg_sample_list = ig_msg_sample(convos[i])
					elif msg_prefix == 'imsg':
						msg_sample_list = imsg_msg_sample(convos[i])
					for msg_t in msg_sample_list:
						print(msg_t)
				elif opt == "y":
					done = True
					print("Adding " + str(convos[i]['name']) + " to people...")
				elif opt == "f":
					filtered_people.append(convos[i]['id'])
					write_filtered(filtered_people, msg_prefix)
					done = True
					print("Filtering " + str(convos[i]['name']) + "...")
				else:
					print(options_str)
		people[convos[i]['id']] = convos[i]['name']
		write_people(msg_prefix, people)

if __name__ == "__main__":
	main()

