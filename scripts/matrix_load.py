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
    #temp = dict
    for guide in dict.copy():
        if dict[guide][2] <= threshold:
            dict.pop(guide, None)
    return dict


def count(dict):
    copies = 0
    for guide in dict:
        copies += dict[guide][2]
    return copies


# Creating dictionaries for guides of the various designs present in algient_NHEJ_final.txt
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

##Loading final matrix data
rep1 = pkl.load(open(workdir + 'NHEJ_rep1_final_matrix.pkl', 'rb'))
rep2 = pkl.load(open(workdir + 'NHEJ_rep2_final_matrix.pkl', 'rb'))
rep3 = pkl.load(open(workdir + 'NHEJ_rep3_final_matrix.pkl', 'rb'))
#mh = pkl.load(open(workdir + 'NHEJ_MH_final_matrix.pkl', 'rb'))
combined = np.vstack([rep1, rep2, rep3])
f2id = pkl.load(open(workdir + 'feature_index_all.pkl', 'rb'))

#Dictionaries for unused features or missing 200mers
missingClasses = {}
missingSequence = {}

final_data = combined
for i in range(len(final_data[:, 6])):
    indel = indelclass(final_data[i, :])
    guide = final_data[i, 6]
    design = final_data[i, 5]
    try:
        id1 = f2id[0][indel]
    except KeyError:
        missingClasses[indel] = [guide]
    else:
        try:
            if design == 'wt':
                table_70k[guide][1][id1] += 1
                table_70k[guide][2] += 1
            elif design == 'mh1':
                table_mh1[guide][1][id1] += 1
                table_mh1[guide][2] += 1
            elif design == 'mh2':
                table_mh2[guide][1][id1] += 1
                table_mh2[guide][2] += 1
            elif design == 'mh3':
                table_mh3[guide][1][id1] += 1
                table_mh3[guide][2] += 1
        except KeyError:
            missingSequence[guide] = [design]


# Normalize events to generate probability density for indel classes
table_70k = normalize(table_70k)
# Remove entries with an insufficient number of reads
th = 10
table_70k = threshold(table_70k, th)

print("Number of guides with more than " + str(th) + " read indel events: " + str(len(table_70k)))
print("Average reads per guide with >10 read indel events: " + str(count(table_70k)/len(table_70k)))
