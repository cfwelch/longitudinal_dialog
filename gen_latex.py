

import subprocess, math, sys, os
#from tqdm import tqdm
from datetime import datetime

from utils import START_TIME, MIN_MESSAGES, MSG_TYPE, liwc_keys, open_for_write, day_of_week_abbr, month_name, season_name, round_sigfigs, short_num_str, tag_set, settings

VERSION_NUMBER = 0.1
TICK_START_YEAR = 2008

def main():
    # initalize style file
    template = None
    with open('latex/nips_2017.sty') as handle:
        template = handle.read()
    template = template.replace('VAR_DATETIME_NOW_STR', datetime.now().strftime("%A %B %-d, %Y @ %-I:%M:%S %p"))

    # write style file
    handle = open_for_write('latex/main.sty')
    handle.write(template)
    handle.close()

    # initalize template
    template = None
    with open('latex/template.tex') as handle:
        template = handle.read()

    # read total message stats
    total_msg_stats = None
    with open('stats/' + settings['prefix'] + '/total_messages_per_person.csv') as handle:
        total_msg_stats = handle.readlines()
    your_msg_total, other_msg_total, your_token_total, other_token_total = [0]*4

    t1parts = total_msg_stats[1].split('\t')
    b1parts = total_msg_stats[-1].split('\t')
    t1_messages, t1_tokens = [int(t1parts[1 + len(MSG_TYPE)]) + int(t1parts[2 + len(MSG_TYPE)]), int(t1parts[3 + len(MSG_TYPE)]) + int(t1parts[4 + len(MSG_TYPE)])]
    b1_messages, b1_tokens = [int(b1parts[1 + len(MSG_TYPE)]) + int(b1parts[2 + len(MSG_TYPE)]), int(b1parts[3 + len(MSG_TYPE)]) + int(b1parts[4 + len(MSG_TYPE)])]
    t1_tpm = t1_tokens * 1.0 / t1_messages if t1_messages > 0 else 0
    b1_tpm = b1_tokens * 1.0 / b1_messages if b1_messages > 0 else 0

    for line in total_msg_stats[1:]:
        parts = line.split('\t')
        your_msg_total += int(parts[1 + len(MSG_TYPE)])
        other_msg_total += int(parts[2 + len(MSG_TYPE)])
        your_token_total += int(parts[3 + len(MSG_TYPE)])
        other_token_total += int(parts[4 + len(MSG_TYPE)])

    # read uniques
    total_uniques = None
    your_utok, other_utok, your_umsg, other_umsg, total_utok, total_umsg = [0]*6
    with open('stats/' + settings['prefix'] + '/total_uniques.txt') as handle:
        total_uniques = handle.readlines()
    for line in total_uniques:
        if line.startswith('tokens me'):
            your_utok = int(line.split(': ')[1])
        elif line.startswith('tokens other'):
            other_utok = int(line.split(': ')[1])
        elif line.startswith('tokens total'):
            total_utok = int(line.split(': ')[1])
        elif line.startswith('messages me'):
            your_umsg = int(line.split(': ')[1])
        elif line.startswith('messages other'):
            other_umsg = int(line.split(': ')[1])
        elif line.startswith('messages total'):
            total_umsg = int(line.split(': ')[1])

    # read hour of day
    hour_of_day_stats = None
    hour_of_day_coords = None
    with open('stats/' + settings['prefix'] + '/hour_of_day_stats.csv') as handle:
        hour_of_day_stats = handle.readlines()
    for line in hour_of_day_stats:
        if line.startswith('All\t'):
            lineparts = line.strip().split('\t')[1:]
            hour_of_day_coords = ''.join(['(' + str(i+1) + ',' + lineparts[i] + ')' for i in range(0, len(lineparts))])

    # read day of week
    day_of_week_stats = None
    day_of_week_coords = None
    with open('stats/' + settings['prefix'] + '/day_of_week_stats.csv') as handle:
        day_of_week_stats = handle.readlines()
    for line in day_of_week_stats:
        if line.startswith('All\t'):
            lineparts = line.strip().split('\t')[1:]
            day_of_week_coords = ''.join(['(' + str(i+1) + ',' + lineparts[i] + ')' for i in range(0, len(lineparts))])

    # read msg per year
    msg_per_year_stats = None
    msg_per_year_coords = None
    with open('stats/' + settings['prefix'] + '/msgs_aggregate.csv') as handle:
        msg_per_year_stats = handle.readlines()
    year_range_ticks = None
    for line in msg_per_year_stats:
        if line.startswith('All Messages\t'):
            msg_per_year_coords = line.strip().split('\t')[1:]
        elif line.startswith('Type\t'):
            lineparts = line.strip().split('\t')[1:]
            year_range_ticks = ','.join(['' if dstr.split(':')[1] == '07' else dstr.split(':')[0][2:] for dstr in lineparts])
    start_index = year_range_ticks.split(',').index(str(TICK_START_YEAR)[2:])
    year_range_ticks = ','.join(year_range_ticks.split(',')[start_index:])
    msg_per_year_coords = msg_per_year_coords[start_index:]
    msg_per_year_coords = ''.join(['(' + str(i+1) + ',' + msg_per_year_coords[i] + ')' for i in range(0, len(msg_per_year_coords))])

    # read response times
    log_resp_time_stats = None
    log_resp_time_coords = ''
    with open('stats/' + settings['prefix'] + '/response_time_bins.txt') as handle:
        log_resp_time_stats = handle.readlines()
    for line in log_resp_time_stats:
        lparts = line.strip().split('\t')
        log_resp_time_coords += '(' + str(math.log(float(lparts[0]))) + ',' + lparts[1] + ')'

    # read month and seasons
    month_season_stats = None
    month_msg_counts = {mnth: 0 for mnth in month_name.values()}
    season_msg_counts = {sea: 0 for sea in season_name.values()}
    with open('stats/' + settings['prefix'] + '/month_season_dists.txt') as handle:
        month_season_stats = handle.readlines()
    for line in month_season_stats:
        lparts = line.strip().split('\t')
        if lparts[0] in month_name.values():
            month_msg_counts[lparts[0]] = int(lparts[1])
        elif lparts[0] in season_name.values():
            season_msg_counts[lparts[0]] = int(lparts[1])

    min_month_messages = min(month_msg_counts.values())
    max_month_messages = max(month_msg_counts.values())
    month_msg_counts = {mnth: round((month_msg_counts[mnth] - min_month_messages) * 50.0 / (max_month_messages - min_month_messages)) + 50 for mnth in month_msg_counts}
    # round them after using them to calculate percentages
    min_month_messages = round_sigfigs(min_month_messages, sigs=2, toward=-1)
    max_month_messages = round_sigfigs(max_month_messages, sigs=2, toward=1)

    # read total group stats
    total_group_stats = None
    total_people = 0
    single_key = list(tag_set.keys())[0] # does not matter which key
    total_group_counts = {tag: {tval: {'messages': 0, 'people': 0} for tval in tag_set[tag]} for tag in tag_set}
    with open('stats/' + settings['prefix'] + '/total_group_stats.csv') as handle:
        total_group_stats = handle.readlines()
    for line in total_group_stats[1:]:
        lparts = line.strip().split('\t')
        if lparts[0] == single_key:
            total_people += int(lparts[2])
        total_group_counts[lparts[0]][lparts[1]]['people'] = float(lparts[4])
        total_group_counts[lparts[0]][lparts[1]]['messages'] = float(lparts[5])

    # read mention graph variables
    mention_graph_nppl, mention_graph_cutoff = [0, 0]
    with open('stats/' + settings['prefix'] + '/mgraph.dot') as handle:
        tlines = handle.readlines()
        mention_graph_cutoff = tlines[0].strip().split('\t')[1]
        mention_graph_nppl = tlines[1].strip().split('\t')[1]

    # replace general values
    template = template.replace('VAR_NAME', settings['my_name'][0])
    template = template.replace('VAR_VERSION_NUMBER', str(VERSION_NUMBER))
    template = template.replace('VAR_MIN_MESSAGES', str(MIN_MESSAGES))
    template = template.replace('VAR_BEGIN_YEAR', str(TICK_START_YEAR))
    template = template.replace('VAR_END_YEAR', str(datetime.now().year))
    template = template.replace('VAR_NUM_PEOPLE', str(total_people))

    # replace total message stat values
    template = template.replace('VAR_YOUR_MESSAGE_TOTAL', '{:,}'.format(your_msg_total))
    template = template.replace('VAR_OTHER_MESSAGE_TOTAL', '{:,}'.format(other_msg_total))
    template = template.replace('VAR_SUM_MESSAGE_TOTAL', '{:,}'.format(your_msg_total + other_msg_total))
    template = template.replace('VAR_YOUR_TOKEN_TOTAL', '{:,}'.format(your_token_total))
    template = template.replace('VAR_OTHER_TOKEN_TOTAL', '{:,}'.format(other_token_total))
    template = template.replace('VAR_SUM_TOKEN_TOTAL', '{:,}'.format(your_token_total + other_token_total))
    template = template.replace('VAR_YOUR_TPM', '{:.2f}'.format(your_token_total * 1.0 / your_msg_total))
    template = template.replace('VAR_OTHER_TPM', '{:.2f}'.format(other_token_total * 1.0 / other_msg_total))
    template = template.replace('VAR_TOTAL_TPM', '{:.2f}'.format((your_token_total + other_token_total) * 1.0 / (your_msg_total + other_msg_total)))
    template = template.replace('VAR_YOUR_UNIQUE_MESSAGES', '{:,}'.format(your_umsg))
    template = template.replace('VAR_OTHER_UNIQUE_MESSAGES', '{:,}'.format(other_umsg))
    template = template.replace('VAR_SUM_UNIQUE_MESSAGES', '{:,}'.format(total_umsg))
    template = template.replace('VAR_YOUR_UNIQUE_TOKENS', '{:,}'.format(your_utok))
    template = template.replace('VAR_OTHER_UNIQUE_TOKENS', '{:,}'.format(other_utok))
    template = template.replace('VAR_SUM_UNIQUE_TOKENS', '{:,}'.format(total_utok))
    template = template.replace('VAR_TOP1_MESSAGE_TOTAL', '{:,}'.format(t1_messages))
    template = template.replace('VAR_TOP1_TOKEN_TOTAL', '{:,}'.format(t1_tokens))
    template = template.replace('VAR_BOTTOM1_MESSAGE_TOTAL', '{:,}'.format(b1_messages))
    template = template.replace('VAR_BOTTOM1_TOKEN_TOTAL', '{:,}'.format(b1_tokens))
    template = template.replace('VAR_TOP1_TPM', '{:.2f}'.format(t1_tpm))
    template = template.replace('VAR_BOTTOM1_TPM', '{:.2f}'.format(b1_tpm))

    # replace time stats
    template = template.replace('VAR_HOUR_OF_DAY', hour_of_day_coords)
    template = template.replace('VAR_DAY_OF_WEEK', day_of_week_coords)
    template = template.replace('VAR_MSG_PER_YEAR', msg_per_year_coords)
    template = template.replace('VAR_YEAR_RANGE_TICKS', year_range_ticks)
    template = template.replace('VAR_YEAR_RANGE_XTICKS', '1,...,' + str(len(year_range_ticks.split(','))))
    template = template.replace('VAR_LOG_RESP_TIME', log_resp_time_coords)
    template = template.replace('VAR_MAX_MONTH_MESSAGES', '{:,}'.format(max_month_messages))
    template = template.replace('VAR_MIN_MONTH_MESSAGES', '{:,}'.format(min_month_messages))

    # month and seasons
    template = template.replace('VAR_JANUARY_COUNT', str(month_msg_counts['january']))
    template = template.replace('VAR_FEBRUARY_COUNT', str(month_msg_counts['february']))
    template = template.replace('VAR_MARCH_COUNT', str(month_msg_counts['march']))
    template = template.replace('VAR_APRIL_COUNT', str(month_msg_counts['april']))
    template = template.replace('VAR_MAY_COUNT', str(month_msg_counts['may']))
    template = template.replace('VAR_JUNE_COUNT', str(month_msg_counts['june']))
    template = template.replace('VAR_JULY_COUNT', str(month_msg_counts['july']))
    template = template.replace('VAR_AUGUST_COUNT', str(month_msg_counts['august']))
    template = template.replace('VAR_SEPTEMBER_COUNT', str(month_msg_counts['september']))
    template = template.replace('VAR_OCTOBER_COUNT', str(month_msg_counts['october']))
    template = template.replace('VAR_NOVEMBER_COUNT', str(month_msg_counts['november']))
    template = template.replace('VAR_DECEMBER_COUNT', str(month_msg_counts['december']))
    template = template.replace('VAR_SPRING_COUNT', short_num_str(season_msg_counts['spring']))
    template = template.replace('VAR_AUTUMN_COUNT', short_num_str(season_msg_counts['fall']))
    template = template.replace('VAR_SUMMER_COUNT', short_num_str(season_msg_counts['summer']))
    template = template.replace('VAR_WINTER_COUNT', short_num_str(season_msg_counts['winter']))

    # group statistics
    template = template.replace('VAR_GENDER_YES_PPL', "{:.1f}".format(total_group_counts['same gender']['yes']['people']))
    template = template.replace('VAR_GENDER_YES_MSGS', "{:.1f}".format(total_group_counts['same gender']['yes']['messages']))
    template = template.replace('VAR_GENDER_NO_PPL', "{:.1f}".format(total_group_counts['same gender']['no']['people']))
    template = template.replace('VAR_GENDER_NO_MSGS', "{:.1f}".format(total_group_counts['same gender']['no']['messages']))
    template = template.replace('VAR_SCHOOL_YES_PPL', "{:.1f}".format(total_group_counts['school']['from school']['people']))
    template = template.replace('VAR_SCHOOL_YES_MSGS', "{:.1f}".format(total_group_counts['school']['from school']['messages']))
    template = template.replace('VAR_SCHOOL_NO_PPL', "{:.1f}".format(total_group_counts['school']['other']['people']))
    template = template.replace('VAR_SCHOOL_NO_MSGS', "{:.1f}".format(total_group_counts['school']['other']['messages']))
    template = template.replace('VAR_WORK_YES_PPL', "{:.1f}".format(total_group_counts['work']['yes']['people']))
    template = template.replace('VAR_WORK_YES_MSGS', "{:.1f}".format(total_group_counts['work']['yes']['messages']))
    template = template.replace('VAR_WORK_NO_PPL', "{:.1f}".format(total_group_counts['work']['no']['people']))
    template = template.replace('VAR_WORK_NO_MSGS', "{:.1f}".format(total_group_counts['work']['no']['messages']))
    template = template.replace('VAR_ROMANTIC_YES_PPL', "{:.1f}".format(total_group_counts['non-platonic relationship']['yes']['people']))
    template = template.replace('VAR_ROMANTIC_YES_MSGS', "{:.1f}".format(total_group_counts['non-platonic relationship']['yes']['messages']))
    template = template.replace('VAR_ROMANTIC_NO_PPL', "{:.1f}".format(total_group_counts['non-platonic relationship']['no']['people']))
    template = template.replace('VAR_ROMANTIC_NO_MSGS', "{:.1f}".format(total_group_counts['non-platonic relationship']['no']['messages']))
    template = template.replace('VAR_AGE_YOUNGER_PPL', "{:.1f}".format(total_group_counts['relative age']['younger']['people']))
    template = template.replace('VAR_AGE_YOUNGER_MSGS', "{:.1f}".format(total_group_counts['relative age']['younger']['messages']))
    template = template.replace('VAR_AGE_OLDER_PPL', "{:.1f}".format(total_group_counts['relative age']['older']['people']))
    template = template.replace('VAR_AGE_OLDER_MSGS', "{:.1f}".format(total_group_counts['relative age']['older']['messages']))
    template = template.replace('VAR_AGE_SAME_PPL', "{:.1f}".format(total_group_counts['relative age']['same']['people']))
    template = template.replace('VAR_AGE_SAME_MSGS', "{:.1f}".format(total_group_counts['relative age']['same']['messages']))
    template = template.replace('VAR_FAMILY_YES_PPL', "{:.1f}".format(total_group_counts['family']['yes']['people']))
    template = template.replace('VAR_FAMILY_YES_MSGS', "{:.1f}".format(total_group_counts['family']['yes']['messages']))
    template = template.replace('VAR_FAMILY_NO_PPL', "{:.1f}".format(total_group_counts['family']['no']['people']))
    template = template.replace('VAR_FAMILY_NO_MSGS', "{:.1f}".format(total_group_counts['family']['no']['messages']))
    template = template.replace('VAR_CCOUNTRY_SAME_PPL', "{:.1f}".format(total_group_counts['same childhood country']['yes']['people']))
    template = template.replace('VAR_CCOUNTRY_SAME_MSGS', "{:.1f}".format(total_group_counts['same childhood country']['yes']['messages']))
    template = template.replace('VAR_CCOUNTRY_OTHER_PPL', "{:.1f}".format(total_group_counts['same childhood country']['no']['people']))
    template = template.replace('VAR_CCOUNTRY_OTHER_MSGS', "{:.1f}".format(total_group_counts['same childhood country']['no']['messages']))
    template = template.replace('VAR_ETHNICITY_SAME_PPL', "{:.1f}".format(total_group_counts['shared ethnicity']['yes']['people']))
    template = template.replace('VAR_ETHNICITY_SAME_MSGS', "{:.1f}".format(total_group_counts['shared ethnicity']['yes']['messages']))
    template = template.replace('VAR_ETHNICITY_OTHER_PPL', "{:.1f}".format(total_group_counts['shared ethnicity']['no']['people']))
    template = template.replace('VAR_ETHNICITY_OTHER_MSGS', "{:.1f}".format(total_group_counts['shared ethnicity']['no']['messages']))

    # LIWC heatmap paths and mention graph
    template = template.replace('VAR_LIWC_GENERAL_PDF_PATH', '../stats/' + settings['prefix'] + '/liwc_general.png')
    template = template.replace('VAR_LIWC_EMOTIONS_PDF_PATH', '../stats/' + settings['prefix'] + '/liwc_emotions.png')
    template = template.replace('VAR_LIWC_CONCERNS_PDF_PATH', '../stats/' + settings['prefix'] + '/liwc_concerns.png')
    template = template.replace('VAR_LIWC_MOTIVATIONS_PDF_PATH', '../stats/' + settings['prefix'] + '/liwc_motivations.png')
    template = template.replace('VAR_MENTION_GRAPH_PATH', '../stats/' + settings['prefix'] + '/mgraph.png')
    template = template.replace('VAR_NPPL_MENTION_GRAPH', mention_graph_nppl)
    template = template.replace('VAR_CUTOFF_MENTION_GRAPH', mention_graph_cutoff)

    # write report
    tex_name = 'gen_' + settings['prefix'] + '_v' + str(VERSION_NUMBER).replace('.', '_')
    handle = open_for_write('latex/' + tex_name + '.tex')
    handle.write(template)
    handle.close()

    # generate pdf
    if os.path.isfile('latex/' + tex_name + '.pdf'):
        os.remove('latex/' + tex_name + '.pdf')
    proc = subprocess.Popen(['latexmk', '-pdf', tex_name + '.tex'], cwd=r'./latex')
    proc.communicate()
    tex_files = [tex_name + '.aux', tex_name + '.fdb_latexmk', tex_name + '.fls', tex_name + '.log']
    for tex_file in tex_files:
        if os.path.isfile('latex/' + tex_file):
            os.remove('latex/' + tex_file)

if __name__ == '__main__':
    main()
