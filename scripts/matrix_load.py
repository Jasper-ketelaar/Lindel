import pickle as pkl
import numpy as np

workdir = '../cwd/'


def indelclass(data):
    indel = ''
    if data[9] == 'del' and data[10] < 2 and data[11] < 30:
        indel = str(data[10]) + '+' + str(data[11])
    elif data[9] == 'ins' and data[11] < 3:
        indel = str(data[11]) + '+' + data[12]
    elif data[9] == 'ins' and data[11] > 2:
        indel = '3'
    return indel


def normalize(dict):
    for guide in dict:
        if dict[guide][2] > 0:
            dict[guide][1] /= dict[guide][2]
    return dict


def threshold(dict, threshold):
    for guide in dict.copy():
        if dict[guide][2] <= threshold:
            dict.pop(guide, None)
    return dict


def corrfilter(dict1, dict2, dict3, thr):
    for gd in {**dict1, **dict2, **dict3}:
        c1, c2, c3 = 0, 0, 0
        temp = 0
        if gd in dict1 and gd in dict2:
            c1 = np.corrcoef(dict1[gd][1], dict2[gd][1])[1, 0]
            temp += 1
        if gd in dict1 and gd in dict3:
            c2 = np.corrcoef(dict1[gd][1], dict3[gd][1])[1, 0]
            temp += 1
        if gd in dict2 and gd in dict3:
            c3 = np.corrcoef(dict2[gd][1], dict3[gd][1])[1, 0]
            temp += 1
        if temp == 0 or (c1 + c2 + c3)/temp < thr:
        #if temp == 0 or max(c1, c2, c3) < thr:
            dict1.pop(gd, None)
            dict2.pop(gd, None)
            dict3.pop(gd, None)

    return dict1, dict2, dict3


def count(dict):
    copies = 0
    for guide in dict:
        copies += dict[guide][2]
    return copies


def indelevents(dict):
    table_70k, table_home, table_mh1, table_mh2, table_mh3, table_May = guidedict()
    for i in range(len(dict[:, 6])):
        indel = indelclass(dict[i, :]) #indelclass provides the name of the indel
        guide = dict[i, 6]
        design = dict[i, 5]
        try:
            id1 = f2id[0][indel]                # Retrieve index of the indel
        except KeyError:                        # If the particular indel type is not found
            if indel not in missingClasses:     # store the indel type with its related guides
                missingClasses[indel] = [guide]
            else:
                missingClasses[indel] = [missingClasses[indel], guide]
            continue
        try:
            if design == 'wt':
                table_70k[guide][1][id1] += 1   # Increases the indel event by 1 (are normalized elsewhere)
                table_70k[guide][2] += 1        # Will increase the total indel events by 1
            elif design == 'mh1':
                table_mh1[guide][1][id1] += 1
                table_mh1[guide][2] += 1
            elif design == 'mh2':
                table_mh2[guide][1][id1] += 1
                table_mh2[guide][2] += 1
            elif design == 'mh3':
                table_mh3[guide][1][id1] += 1
                table_mh3[guide][2] += 1
        except KeyError:                        # If there is no reference for the guide
            missingSequence[guide] = [design]   # store it together with its design type
    return table_70k

# Creating dictionaries for guides of the various designs present in algient_NHEJ_final.txt
def guidedict():
    ref = open(workdir + 'algient_NHEJ_guides_final.txt')
    table_home = {}
    table_70k = {}
    table_mh1 = {}
    table_mh2 = {}
    table_mh3 = {}
    table_May = {}

    for line in ref:
        seq, label = line.rstrip('\r\n').split('\t')
        guide = seq[20:40]

        if label == '70k seq design' and guide not in table_70k:
            table_70k[guide] = [seq, np.zeros(557), 0]
        elif label == 'homing_design' and guide not in table_mh1:
            table_home[guide] = [seq, np.zeros(557), 0]
        elif label == 'mh_design_1' and guide not in table_mh1:
            table_mh1[guide] = [seq, np.zeros(557), 0]
        elif label == 'mh_design_2' and guide not in table_mh2:
            table_mh2[guide] = [seq, np.zeros(557), 0]
        elif label == 'mh_design_3' and guide not in table_mh3:
            table_mh3[guide] = [seq, np.zeros(557), 0]
        elif label == 'Maydata' and guide not in table_May:
            table_May[guide] = [seq, np.zeros(557), 0]
    return table_70k, table_home, table_mh1, table_mh2, table_mh3, table_May


# Loading final matrix data
rep1 = pkl.load(open(workdir + 'NHEJ_rep1_final_matrix.pkl', 'rb'))
rep2 = pkl.load(open(workdir + 'NHEJ_rep2_final_matrix.pkl', 'rb'))
rep3 = pkl.load(open(workdir + 'NHEJ_rep3_final_matrix.pkl', 'rb'))
# mh = pkl.load(open(workdir + 'NHEJ_MH_final_matrix.pkl', 'rb'))
# combined = np.vstack([rep1, rep2, rep3])
f2id = pkl.load(open(workdir + 'feature_index_all.pkl', 'rb'))

# Dictionaries for missing features and missing 200mers
missingClasses = {}
missingSequence = {}

# Dictionaries for data
r1 = indelevents(rep1)
r2 = indelevents(rep2)
r3 = indelevents(rep3)

# Normalize events to generate probability density for indel classes
# Remove entries with an insufficient number of reads
th = 10
r1 = normalize(threshold(r1, th))
r2 = normalize(threshold(r2, th))
r3 = normalize(threshold(r3, th))

# Remove guides for which the three reps do not sufficiently correlate
cth = 0.75
print(len({**r1, **r2, **r3}))
r1, r2, r3 = corrfilter(r1, r2, r3, cth)
print(len({**r1, **r2, **r3}))

# Merges the normalized, correlation filtered reps.
merged = {}
for gd in {**r1, **r2, **r3}:
    a = 1


#print(len(merged))

def targetseq(designseq):
    guide = designseq[20:40]
    id = 100 + designseq[100:].index(guide)
    target = designseq[(id-13):(id+52)]
    return target
