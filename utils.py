

import datetime, random, socket, time, math, json, pytz, os

#from normalizer import expand_text, norms
from urllib.parse import urlparse
from termcolor import colored
from pprint import pprint

dir_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(dir_path + "/logs", exist_ok=True)


#######################################################
#             User specific variables                 #
#######################################################

CONFIG_FILE = 'charlie.cfg'

settings = {}
with open('config/' + CONFIG_FILE) as handle:
	lines = handle.readlines()
	for line in lines:
		tline = line.strip()
		if tline.startswith('#') or tline == '':
			continue
		tline = tline.split('=')
		assert len(tline) == 2
		var_name = tline[0].strip()
		val_list = []
		for part in tline[1].strip().split(','):
			tpart = part.strip()
			if tpart == 'None':
				continue
			else:
				tpart = tpart[1:] if tpart.startswith('\'') or tpart.startswith('\"') else tpart
				tpart = tpart[:-1] if tpart.endswith('\'') or tpart.endswith('\"') else tpart
				if tpart != '':
					val_list.append(tpart)
		settings[var_name] = val_list

# Flatten lists
settings['GOOGLE_FTYPE'] = settings['GOOGLE_FTYPE'][0]
settings['FACEBOOK_FTYPE'] = settings['FACEBOOK_FTYPE'][0]
settings['prefix'] = settings['prefix'][0]

settings['PEOPLE_FOLDER'] = 'people/' + settings['prefix']
settings['DATA_DIR'] = 'data/' + settings['prefix']
settings['DATA_MERGE_DIR'] = 'data_merged/' + settings['prefix']
settings['LIWC_PATH'] = dir_path + '/LIWC.2015.all'

# print(settings)

#######################################################
#       Constants for Analysis and Prediction         #
#######################################################

MSG_TYPE = {"SMS": 0, "FB": 1, "GH": 2, "IMSG": 3, "AIM": 4, "IG": 5}
MSG_TYPE_PATHS = {k: None for k in MSG_TYPE.keys()}
MSG_TYPE_PATHS['GH'] = settings['GOOGLE_PATH']
MSG_TYPE_PATHS['FB'] = settings['FACEBOOK_PATH']
MSG_TYPE_PATHS['IMSG'] = settings['IMESSAGE_PATH']
MSG_TYPE_PATHS['IG'] = settings['INSTAGRAM_PATH']

NUMBER_OF_PEOPLE = 150 # Affects the construction of graphs for graph features

URL_PATTERN = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

tag_set = {"same gender": ["yes", "no"], "family": ["yes", "no"], "school": ["from school", "other"], "non-platonic relationship": ["yes", "no"], "relative age": ["younger", "older", "same"], "work": ["yes", "no"], "shared ethnicity": ["yes", "no"], "same childhood country": ["yes", "no", "unknown"]}

tag_description = {"same gender": "Answer yes if this person has the same gender as you.", "family": "Answer yes if this person is related to you.", "school": "Answer 'from school' if you met this person at a school you were attending.", "non-platonic relationship": "Answer yes if your relationship with this person was ever not platonic, e.g. girlfriend/boyfriend, spouse, person you went on a date with.", "relative age": "Answer 'younger' if this person is younger than you are, and 'older' if this person is older than you are. Answer 'same' if this persons age is within 1.5 years of your age.", "work": "Answer yes if you met this person at a place you working at the time you met them.", "shared ethnicity": "Answer yes if you share an ethnic background with this person.", "same childhood country": "Answer yes if this person grew up in the same country as you."}

FAM_IND, NPR_IND, RA_IND, SCC_IND, SG_IND, SCH_IND, SETH_IND, WORK_IND = range(8)
uax_set = [FAM_IND, NPR_IND, RA_IND, SCC_IND, SG_IND, SCH_IND, WORK_IND]
uax_names = ["family", "romantic", "age", "country", "gender", "school", "work"]

os.makedirs(dir_path + "/" + settings['DATA_DIR'], exist_ok=True)
START_TIME = datetime.datetime(year=2006, month=11, day=1)
DEFAULT_TIMEZONE = pytz.timezone('US/Eastern')

label_set = ["yes", "haha", "okay", "oh", "i do not know", "hm", "no", "nice", "cool", "hi", "?", "what is up", "thanks"]
time_label_size = {"short": 90, "10min": 600, "day": 3600*24, "longer": 315576000} #"hour": 3600
time_labels = ["short", "10min", "day", "longer"] #"hour"

SPTOK_NU = '<NU>'
SPTOK_ME = '<ME>'
SPTOK_OTHER = '<OTHER>'
SPTOK_UNK = '<UNK>'
SPTOK_NL = '<NL>'
SPTOK_EOS = '<EOS>'
SPTOK_SOS = '<SOS>'

TOP_WORDS = 150
MIN_MESSAGES = 100
EMBEDDING_SIZE = 300
EMBEDDING_LOCATION = '/path/to/embeddings/'

# Filter off-the-record communications
OTR_STRS = ['youremail@place.com/', ' has requested an Off-the-Record private conversation <','https://otr.cypherpunks.ca/','>.  However, you do not have a plugin to support that.','https://otr.cypherpunks.ca/',' for more information.']
OTR_CONTEXT = ['See ']

month_to_season = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
season_name = {0: "winter", 1: "spring", 2: "summer", 3: "fall"}
month_name = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june", 7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}
day_of_week_abbr = {0: 'M', 1: 'Tu', 2: 'W', 3: 'Th', 4: 'F', 5: 'Sa', 6: 'Su'}

liwc_keys = ['ACHIEV', 'ADJ', 'ADVERB', 'AFFECT', 'AFFILIATION', 'ANGER', 'ANX', 'ARTICLE', 'ASSENT', 'AUXVERB', 'BIO', 'BODY', 'CAUSE', 'CERTAIN', 'COGPROC', 'COMPARE', 'CONJ', 'DEATH', 'DIFFER', 'DISCREP', 'DRIVES', 'FAMILY', 'FEEL', 'FEMALE', 'FILLER', 'FOCUSFUTURE', 'FOCUSPAST', 'FOCUSPRESENT', 'FRIEND', 'FUNCTION', 'HEALTH', 'HEAR', 'HOME', 'I', 'INFORMAL', 'INGEST', 'INSIGHT', 'INTERROG', 'IPRON', 'LEISURE', 'MALE', 'MONEY', 'MOTION', 'NEGATE', 'NEGEMO', 'NETSPEAK', 'NONFLU', 'NUMBER', 'PERCEPT', 'POSEMO', 'POWER', 'PPRON', 'PREP', 'PRONOUN', 'QUANT', 'RELATIV', 'RELIG', 'REWARD', 'RISK', 'SAD', 'SEE', 'SEXUAL', 'SHEHE', 'SOCIAL', 'SPACE', 'SWEAR', 'TENTAT', 'THEY', 'TIME', 'VERB', 'WE', 'WORK', 'YOU']

lsm_keys = ['ADVERB', 'ARTICLE', 'AUXVERB', 'CONJ', 'IPRON', 'NEGATE', 'PPRON', 'PREP', 'QUANT']#, 'INFORMAL', 'VERB', 'ADJ', 'COMPARE', 'INTERROG', 'NUMBER']


#######################################################
#               Useful functions                      #
#######################################################

def short_num_str(number):
	places = math.floor(math.log10(number)) + 1
	if places <= 3: # hundreds or lower -- no append
		return str(number)
	elif places <= 6: # thousands
		return ("{:,." + str(6-places) + "f}").format(number/1000) + "k"
	elif places <= 9: # millions
		return ("{:,." + str(9-places) + "f}").format(number/1000000) + "m"
	else:
		return "{:,.0f}".format(number/1000000) + "m"

def round_sigfigs(number, sigs=1, toward=0):
	if toward == 0:
		return round(number, -int(math.floor(math.log10(abs(number)))) + (sigs-1))
	else:
		divisor = 10**(math.floor(math.log10(number)) - (sigs-1))
		if toward > 0:
			return math.ceil(number/divisor)*divisor
		else: #toward<0
			return math.floor(number/divisor)*divisor

def make_user_vector(people, name, as_parts=False):
	ret_vec = []
	if name in people:
		keys = list(tag_set.keys())
		keys.sort()
		for i in range(len(keys)):
			tval_list = [0]*len(tag_set[keys[i]])
			tval_list[tag_set[keys[i]].index(people[name][keys[i]])] = 1
			if as_parts:
				ret_vec.append(tval_list)
			else:
				ret_vec.extend(tval_list)
		if not as_parts:
			assert len(ret_vec) == sum([len(tag_set[tag]) for tag in tag_set])
	else:
		print("Name is: " + str(name))
		assert name in f_my_name
	return ret_vec

def uv_to_parts(vector):
	return [vector[0:2], vector[2:4], vector[4:7], vector[7:10], vector[10:12], vector[12:14], vector[14:16], vector[16:18]]

def read_people_file(f_name):
	people = {}
	if os.path.exists(settings['PEOPLE_FOLDER'] + "/" + f_name):
		with open(settings['PEOPLE_FOLDER'] + "/" + f_name) as handle:
			people = json.load(handle)
	return people

def write_people(prefix, people_dict):
	handle = open_for_write(settings['PEOPLE_FOLDER'] + "/" + prefix + "_people")
	json.dump(people_dict, handle)

def write_filtered(filtered_people, prefix):
	sfp = set(filtered_people)
	handle = open_for_write(settings['PEOPLE_FOLDER'] + "/" + prefix + "_filtered_people")
	for person in sfp:
		handle.write(person + "\n")

def read_filtered(prefix):
	filtered_people = []
	# Read filtered people
	if os.path.exists(settings['PEOPLE_FOLDER'] + "/" + prefix + "_filtered_people"):
		with open(settings['PEOPLE_FOLDER'] + "/" + prefix + "_filtered_people") as handle:
			for line in handle.readlines():
				filtered_people.append(line.strip())
	return filtered_people

def gprint(msg, logname, error=False, important=False):
	tmsg = msg if not important else colored(msg, "cyan")
	tmsg = tmsg if not error else colored(msg, "red")
	st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
	cmsg = str(st) + ": " + str(tmsg)
	tmsg = str(st) + ": " + str(msg)
	print(cmsg)
	log_file = open(dir_path + "/logs/" + logname + ".log", "a")
	log_file.write(tmsg + "\n")
	log_file.flush()
	log_file.close()

# Opens a file handle but makes the needed directories
def open_for_write(filename, binary=False):
	mkpath = "/".join(filename.split("/")[:-1])
	os.makedirs(mkpath, exist_ok=True)
	return open(filename, "w" + ("b" if binary else ""))

def get_domain(inurl):
	retval = inurl
	if not inurl.startswith("www."):
		t_val = urlparse(inurl).netloc
		if t_val != "":
			retval = t_val
	#print("rvaL: " + retval)
	if "://" in retval:
		retval = retval[retval.index("://")+3:]
	#print("rvaL: " + retval)
	if "www." in retval:
		retval = retval[retval.index("www.")+4:]
	#print("rvaL: " + retval)
	if "/" in retval:
		retval = retval[:retval.index("/")]
	#print("rvaL: " + retval)
	return retval
