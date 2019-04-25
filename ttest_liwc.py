

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(rc={'figure.facecolor': 'cornflowerblue'}) #'axes.facecolor': 'cornflowerblue'

import matplotlib.ticker as tkr
formatter = tkr.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

from tqdm import tqdm
from scipy.stats import ttest_ind #ttest_ind_from_stats
from utils import START_TIME, settings, liwc_keys, read_people_file, tag_set, open_for_write

def ttest_make_heat_maps(save_as_fig=False):
    valid_people = read_people_file('tagged_people')
    liwc_people = None
    with open('stats/' + settings['prefix'] + '/liwc_person.csv') as handle:
        liwc_people = handle.readlines()

    liwc_lines = [ll.strip() for ll in liwc_people]
    #ppl = {name: {lkey: 0 for lkey in liwc_keys} for name in valid_people}
    #group_key = 'same gender' #'school'
    #group_key = 'same childhood country'
    total_sets = ["WORK", "LEISURE", "HOME", "MONEY", "RELIG", "DEATH", \
                "POSEMO", "NEGEMO", "ANX", "ANGER", "SAD", \
                "SOCIAL", "FAMILY", "FRIEND", "MALE", "FEMALE", \
                "COGPROC", "INSIGHT", "CAUSE", "DISCREP", "TENTAT", "CERTAIN", "DIFFER", \
                "BIO", \
                #, "BODY", "HEALTH", "SEXUAL", "INGEST"
                "PERCEPT", \
                #, "SEE", "HEAR", "FEEL"
                "DRIVES", "AFFILIATION", "ACHIEV", "POWER", "REWARD", "RISK", \
                #"FOCUSPRESENT", "FOCUSPAST", "FOCUSFUTURE", \
                "RELATIV", \
                #, "MOTION", "SPACE", "TIME"
                "INFORMAL"]#, "SWEAR", "NETSPEAK", "ASSENT", "NONFLU", "FILLER"]
    #liwc_cats = list(set(liwc_keys).difference(set(liwc_cats)))

    #"SOCIAL": "Social All", "COGPROC": "Cognitive All"
    group_names = {"ANGER": "a_ANGER", "ANX": "b_ANX", "POSEMO": "c_POSEMO", "SAD": "d_SAD", \
                    "BIO": "e_BIO", "COGPROC": "f_COGPROC", "RELATIV": "g_RELATIV", \
                    "ACHIEV": "h_ACHIEV", "AFFILIATION": "i_AFFILIATION", "REWARD": "j_REWARD", "RISK": "k_RISK", "POWER": "l_POWER", \
                    "DEATH": "m_DEATH", "HOME": "n_HOME", "LEISURE": "o_LEISURE", "MONEY": "p_MONEY", "RELIG": "q_RELIG", "WORK": "r_WORK"}

    #"SOCIAL"] "AFFECT", "PERCEPT", "BIO", "RELATIV", 
    liwc_cats = ["ANGER", "ANX", "POSEMO", "SAD", \
                #"DRIVES", "COGPROC", "CONCERNS", "FUNCTION", "GRAMMAR", "NEGEMO", "INFORMAL", \
                #"INSIGHT", "CAUSE", "DISCREP", "TENTAT", "CERTAIN", "DIFFER", \
                "BIO", "COGPROC", "RELATIV", \
                "ACHIEV", "AFFILIATION", "REWARD", "RISK", "POWER", \
                "DEATH", "HOME", "LEISURE", "MONEY", "RELIG", "WORK"]

    # all high level categories
    #liwc_cats = ["SOCIAL", "AFFECT", "PERCEPT", "BIO", "RELATIV", "CONCERNS", "COGPROC", "DRIVES"]

    liwc_combins = ["GRAMMAR", "CONCERNS"]
    liwc_mapper = {"GRAMMAR": ["ADJ", "VERB", "COMPARE", "INTERROG", "NUMBER", "QUANT"], "CONCERNS": ["WORK", "LEISURE", "HOME", "MONEY", "RELIG", "DEATH"]}
    liwc_rmapper = {"ADJ": "GRAMMAR", "VERB": "GRAMMAR", "COMPARE": "GRAMMAR", "INTERROG": "GRAMMAR", "NUMBER": "GRAMMAR", "QUANT": "GRAMMAR", "WORK": "CONCERNS", "LEISURE": "CONCERNS", "HOME": "CONCERNS", "MONEY": "CONCERNS", "RELIG": "CONCERNS", "DEATH": "CONCERNS"}

    #fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
    #ax = axn.flat
    #iplot = 0

    indicies = []
    all_trues = {lval: [] for lval in liwc_cats}
    all_pvals = {lval: [] for lval in liwc_cats}
    pval_sym = {lval: [] for lval in liwc_cats}
    the_key_set = [qq for qq in tag_set.keys() if qq != "shared ethnicity" and qq != "relative age"]
    the_key_set.extend(["relative age0", "relative age1", "relative age2"])
    for group_key in the_key_set:

        liwc_order = liwc_lines[0].split('\t')
        if group_key.startswith("relative age"):
            the_rlind = int(group_key[-1])
            other_rlinds = [0,1,2]
            other_rlinds.remove(the_rlind)
            group_key = group_key[:-1]
            pos_val = tag_set[group_key][the_rlind]
            neg_val = [tag_set[group_key][qk] for qk in other_rlinds]
        else:
            pos_val = tag_set[group_key][0]
            neg_val = tag_set[group_key][1]
        #assert len(tag_set[group_key]) == 2

        if group_key.startswith("relative age"):
            indicies.append("relative age: " + str(pos_val) + "-other")
        else:
            if group_key == "non-platonic relationship":
                indicies.append("romantic: " + str(pos_val) + "-" + str(neg_val))
            elif group_key == "same childhood country":
                indicies.append("childhood country: USA-other")
            elif group_key == "school":
                indicies.append("school: yes-no")
            else:
                indicies.append(group_key + ": " + str(pos_val) + "-" + str(neg_val))

        bonferroni = len(liwc_cats)
        print("Len LIWC keys: " + str(bonferroni))
        ### true diff
        group_values = {t_val: {lkey: 0 for lkey in liwc_keys} for t_val in tag_set[group_key]}
        group_vlist = {t_val: {lkey: [] for lkey in liwc_keys} for t_val in tag_set[group_key]}

        for t_val in tag_set[group_key]:
            group_values[t_val]['words'] = 0
            for llcb in liwc_combins:
                group_vlist[t_val][llcb] = []

        for line in liwc_lines[1:]:
            parts = line.split('\t')
            if parts[0] in settings['my_name']:
                continue
            t_val = valid_people[parts[0]][group_key]
            group_values[t_val]['words'] += int(parts[1])
            offset = 2
            for i in range(offset, len(parts)):
                group_values[t_val][liwc_order[i]] += int(parts[i])
                # do the q-val merge here
                vlist_key = liwc_order[i] if liwc_order[i] not in liwc_combins else liwc_rmapper[liwc_order[i]]
                group_vlist[t_val][vlist_key].append(int(parts[i])*1.0/int(parts[1]) if int(parts[1]) > 0 else 0)

        trues = {}
        print(("="*30) + "\nGroup Key: " + group_key + "\n" + ("="*30))
        #for t_val in tag_set[group_key]:
        print("Pos value is: " + pos_val)
        print("Neg value is: " + str(neg_val))
        for lval in liwc_cats:
            q_val_set = liwc_mapper[lval] if lval in liwc_combins else [lval]
            pos_normed = sum([group_values[pos_val][q_val] for q_val in q_val_set]) * 1.0 / group_values[pos_val]['words'] if group_values[pos_val]['words'] > 0 else 0
            neg_normed = 0.0
            for q_val in q_val_set:
                if group_key.startswith("relative age"):
                    neg_normed += sum([group_values[qp][q_val] for qp in neg_val]) * 1.0 / sum([group_values[qp]['words'] for qp in neg_val]) if sum([group_values[qp]['words'] for qp in neg_val]) > 0 else 0
                else:
                    neg_normed += group_values[neg_val][q_val] * 1.0 / group_values[neg_val]['words'] if group_values[neg_val]['words'] > 0 else 0
            print('\t' + lval + ': ' + str(pos_normed - neg_normed))
            trues[lval] = pos_normed - neg_normed


        # Calculate the p-values
        for lval in liwc_cats:
            pos_values = np.array(group_vlist[pos_val][lval])
            neg_values = []
            if group_key.startswith("relative age"):
                for qp in neg_val:
                    neg_values.extend(group_vlist[qp][lval])
            else:
                neg_values = group_vlist[neg_val][lval]
            neg_values = np.array(neg_values)

            #print("\npos values are: " + str(pos_values))
            #print("neg values are: " + str(neg_values))
            ttest_t, ttest_p = ttest_ind(pos_values, neg_values, equal_var=False)

            other_tested = 0#11
            adj_p = ttest_p * (bonferroni + other_tested) * len(the_key_set)
            #print(" -- " + str(ttest_p) + " times " + str(bonferroni + other_tested) + " time " + str(len(the_key_set)) + " equals " + str(adj_p))

            lval_str = (lval + " "*10)[:10]
            print("ttest_ind pvalue for " + lval_str + " with bonferroni(" + str((bonferroni + other_tested) * len(the_key_set)) + "): " + "{:.4f}".format(ttest_p) + "\tadjusted to: " + "{:.4f}".format(adj_p) + " with t-value: " + "{:.4f}".format(ttest_t))

            all_trues[lval].append(trues[lval])
            #all_pvals[lval].append(adj_p)
            all_pvals[lval].append(float(ttest_p))
            pval_sym[lval].append("?")

    ################################
    # Holm-Bonferroni correction -- across lvals -- is wrong
    #ordered_pval_inds = sorted(range(len(all_pvals)), key=lambda lamdakey: all_pvals[lamdakey])
    for lval in all_pvals:
        ordered_pvals = sorted(all_pvals[lval])

        point01thresh = -1
        point05thresh = -1
        point10thresh = -1
        for ii in range(len(ordered_pvals)):
            adjp = ordered_pvals[ii] * (len(ordered_pvals) - ii)
            if adjp > 0.10 and point10thresh == -1:
                point10thresh = ii
            if adjp > 0.05 and point05thresh == -1:
                point05thresh = ii
            if adjp > 0.01 and point01thresh == -1:
                point01thresh = ii

        print(lval + " 0.01 threshold: " + str(point01thresh))
        print(lval + " 0.05 threshold: " + str(point05thresh))
        print(lval + " 0.10 threshold: " + str(point10thresh))

        for ii in range(len(all_pvals[lval])):
            ord_ind_p = ordered_pvals.index(all_pvals[lval][ii])
            if ord_ind_p < point01thresh:
                pval_sym[lval][ii] = "***"
            elif ord_ind_p < point05thresh:
                pval_sym[lval][ii] = "**"
            elif ord_ind_p < point10thresh:
                pval_sym[lval][ii] = "*"
            else:
                pval_sym[lval][ii] = ""
    ################################


    ############# SHOW STUFF
    do_them_all_together = False
    #cmap = 'YlGnBu'
    #cmap = sns.diverging_palette(240, 10, sep=1, as_cmap=True)
    cmap = sns.diverging_palette(275, 132, sep=1, as_cmap=True)
    vmin, vmax = [-0.010, 0.010]

    #if do_them_all_together:
    dq = {group_names[lval]: all_trues[lval] for lval in liwc_cats}
    df = pd.DataFrame(index=indicies, data=dq)

    ax = sns.heatmap(df, annot=True, fmt=".2f", square=1, linewidth=1., xticklabels=1, cmap=cmap, vmin=-0.015, vmax=0.015)#mask=(abs(df)<0.0001)

    counter = 0
    for text in ax.texts:
        #text.set_text(str(counter))
        index_group = int(counter / len(liwc_cats))
        liwc_cat = counter % len(liwc_cats)
        text.set_text(pval_sym[liwc_cats[liwc_cat]][index_group])
        """if float(all_pvals[liwc_cats[liwc_cat]][index_group]) < 0.01:
            text.set_text("***")
        elif float(all_pvals[liwc_cats[liwc_cat]][index_group]) < 0.05:
            text.set_text("**")
        elif float(all_pvals[liwc_cats[liwc_cat]][index_group]) < 0.1:
            text.set_text("*")
        else:
            text.set_text("")"""
        counter += 1
    if not save_as_fig:
        plt.show()
    #else:


    grp1, grp1p = [["ANGER", "ANX", "POSEMO", "SAD"], ["Anger", "Anxiety", "Positive", "Sad"]]
    grp2, grp2p = [["BIO", "COGPROC", "RELATIV"], ["Biological", "Cognitive", "Relativity"]]
    grp3, grp3p = [["ACHIEV", "AFFILIATION", "REWARD", "RISK", "POWER"], ["Achievement", "Affiliation", "Reward", "Risk", "Power"]]
    grp4, grp4p = [["DEATH", "HOME", "LEISURE", "MONEY", "RELIG", "WORK"], ["Death", "Home", "Leisure", "Money", "Religion", "Work"]]
    grps = [grp1, grp2, grp3, grp4]
    grps_p = [grp1p, grp2p, grp3p, grp4p]
    grp_names = ["emotions", "general", "motivations", "concerns"]
    grp_index = 0

    print(pval_sym)

    for grp in grps:
        #dq = {lval: all_trues[lval] for lval in liwc_cats}
        dq = {group_names[lval]: all_trues[lval] for lval in grp}
        df = pd.DataFrame(index=indicies, data=dq)

        # vmin=-0.015, vmax=0.015
        fig, ax = plt.subplots()
        xticks = grps_p[grp_index]
        if grp_index == 1:
            yticks = ["Same Gender", "Family", "School", "Romantic", "Work", "Same Country", "Younger", "Older", "Same Age"]
        else:
            yticks = False

        xticks, yticks = [False]*2 # Force none
        ax = sns.heatmap(df, annot=True, fmt=".2f", square=1, linewidth=1., xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar_kws={'format':formatter}, center=0)
        #ax.set_facecolor((0.7226, 0.8008, 0.8945))

        counter = 0
        print("grp is...: " + str(grp))
        for text in ax.texts:
            #text.set_text(str(counter))
            index_group = int(counter / len(grp))
            liwc_cat = counter % len(grp)
            print("index_group: " + str(index_group))
            print("liwc_cat: " + str(liwc_cat))
            print("grp[liwc_cat]: " + str(grp[liwc_cat]))
            print("pval_sym[grp[liwc_cat]]: " + str(pval_sym[grp[liwc_cat]]))
            print("pval_sym[grp[liwc_cat]][index_group]: " + str(pval_sym[grp[liwc_cat]][index_group]))
            text.set_text(pval_sym[grp[liwc_cat]][index_group])
            counter += 1

        #ax.set_title('LIWC normalized count difference between groups x10^4')
        #iplot += 1

        #fig.tight_layout()
        if save_as_fig:
            plt.savefig('stats/' + settings['prefix'] + '/liwc_' + grp_names[grp_index] + '.png', bbox_inches='tight', transparent=True)
            plt.clf()
        else:
            plt.show()
        grp_index += 1

if __name__ == "__main__":
    ttest_make_heat_maps()
