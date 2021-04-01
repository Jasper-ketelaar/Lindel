import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

workdir = '../cwd/'


# Returns the name of the class
def indelclass(data):
    indel = 'place holder'
    if data[9] == 'del': # and data[11] < 30: # and data[10] < 2:
        indel = str(data[10]) + '+' + str(data[11])
    elif data[9] == 'ins' and data[11] < 3:
        indel = str(data[11]) + '+' + data[12]
    elif data[9] == 'ins' and data[11] > 2:
        indel = '3'
    return indel


# This function normalizes the indel events (thereby becoming a class probability distribution)
def normalize(dict):
    for guide in dict:
        if dict[guide][2] > 0:
            dict[guide][1] /= dict[guide][2]
    return dict


# Removes guides from the dictionaries if they have insufficient reads
def threshold(dict, threshold):
    for guide in dict.copy():
        if dict[guide][2] < threshold:
            dict.pop(guide, None)
    return dict


# Filters out entries of the dictionaries if the average correlation is below the threshold
def corrfilter(dict1, dict2, dict3, cthr):
    for gd in {**dict1, **dict2, **dict3}:
        c1, c2, c3 = 0, 0, 0
        #c1, c2, c3 = 1, 1, 1
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
        if temp == 0 or (c1 + c2 + c3)/temp < cthr: # Average correlation implementation
        # if temp == 0 or min(c1, c2, c3) < cthr:
            dict1.pop(gd, None)
            dict2.pop(gd, None)
            dict3.pop(gd, None)
    return dict1, dict2, dict3


# Returns the total number of reads of a dictionary
def count(dict):
    copies = 0
    for guide in dict:
        copies += dict[guide][2]
    return copies


# Assigns indel events from final matrix data to the guides of the various designs
def indelevents(array):
    table_70k, table_home, table_mh1, table_mh2, table_mh3, table_May = guidedict()
    for i in range(len(array[:, 6])):
        indel = indelclass(array[i, :])          # indelclass provides the name of the indel
        guide = array[i, 6]
        design = array[i, 5]
        try:
            id1 = f2id[0][indel]                # Retrieve index of the indel
        except KeyError:                        # If the particular indel type is not found
            if indel != 'place holder':
                missingClasses.append(indel)
                missingClassSequences.append(guide)
            continue                            # proceed to the next iteration
        try:
            if design == 'wt':
                table_70k[guide][1][id1] += 1   # Increases the indel event by 1 (are normalized elsewhere)
                table_70k[guide][2] += 1        # Will increase the total indel events by 1
            elif design == 'mh1':
                table_mh1[guide][1][id1] += 1   # Not knowing if the MH final matrix, containing guides with multiple
                table_mh1[guide][2] += 1        # designs, would be useful, its guides were separately stored.
            elif design == 'mh2':
                table_mh2[guide][1][id1] += 1
                table_mh2[guide][2] += 1
            elif design == 'mh3':
                table_mh3[guide][1][id1] += 1
                table_mh3[guide][2] += 1
        except KeyError:                        # If there is no reference for the guide
            missingSequence.append(guide)       # store the guide and verify if its design is 70k
            QualityControl.append(array[i, 1].replace('-', '').__contains__('CGACTGCTAACGTTATCAAC'))
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
# microh = pkl.load(open(workdir + 'NHEJ_MH_final_matrix.pkl', 'rb'))
# combined = np.vstack([rep1, rep2, rep3])

# Loading feature look-up table
f2id = pkl.load(open(workdir + 'feature_index_all.pkl', 'rb'))

# Lists for missing classes, the corresponding guide, and missing 200mers
missingClasses = []
missingClassSequences = []
missingSequence = []
QualityControl = []

# Dictionaries for data
r1 = indelevents(rep1)
r2 = indelevents(rep2)
r3 = indelevents(rep3)
# mh = indelevents(microh)
total_guides = len({**r1, **r2, **r3})

# Remove entries with an insufficient number of reads
th = 10
r1 = threshold(r1, th)
r2 = threshold(r2, th)
r3 = threshold(r3, th)
# mh = threshold(mh, th)
threshold_guides = len({**r1, **r2, **r3})

# Remove guides for which the three replicates do not sufficiently correlate
cth = 0.75
r1, r2, r3 = corrfilter(r1, r2, r3, cth)

# Normalizes data
# r1 = normalize(r1)
# r2 = normalize(r2)
# r3 = normalize(r3)

# Merges the correlation filtered replicates
merged = {}
for gd in {**r1, **r2, **r3}:
    count = 0
    seq = ''
    events = np.zeros(557)
    if gd in r1:
        seq = r1[gd][0]
        events += r1[gd][1]
        count += r1[gd][2]
    if gd in r2:
        seq = r2[gd][0]
        events += r2[gd][1]
        count += r2[gd][2]
    if gd in r3:
        seq = r3[gd][0]
        events += r3[gd][1]
        count += r3[gd][2]
    # if gd in mh:
    #    events += mh[gd][1]
    #    count += mh[gd][2]
    merged[gd] = [seq, events, count]

# Normalizes the class events to a class probability distribution
merged = normalize(merged)

### Evaluating the data ###
# Not all guides had a corresponding reference sequence in the algient txt file
print('The algient file contains ' + str(len(guidedict()[0])) + ' reference sequences.')
print(str(len(set(missingSequence))) + ' unique guide sequences (with ' + str(len(missingSequence)) + ' measured indels)\
 had no reference in the algient file.')
# print(missingSequence)

# Some indels are not included in the model
uniqueMissingClasses = set(missingClasses)
print(str(len(uniqueMissingClasses)) + ' measured indel types were not included in the model\'s classes\
: ' + str(uniqueMissingClasses) + ', ignoring ' + str(len(missingClasses)) + ' measured indels.')

# Inspecting our output
print('\nThe total data contains ' + str(total_guides) + ' sequences. After thresholding,\
 ' + str(threshold_guides) + ' sequences remain.')
print('After removing insufficiently correlating replicates, ' + str(len(merged)) + ' sequences remain.')
# Loads the test set provided by the authors
f = open(workdir + "Lindel_test.txt", "r")

### Evaluating the output ###
# Extracts the guide sequence and class distributions
test = {}
for line in f:
    temp = line.split('\t')
    guide = temp[0]
    # features = [float(item) for item in temp[1:3034]]
    classes = [float(item) for item in temp[3034:]]
    test[guide] = classes

# Compares the merged data to the test set of the authors
cor = []
absError = []
temp = 0
for gd in test:
    try:
        cor.append(np.corrcoef(test[gd], merged[gd][1])[0, 1])
        absError.append(sum(abs(test[gd] - merged[gd][1])))
    except KeyError:
        if gd in missingSequence:
            temp += 1
        continue

#print(str(sum(QualityControl)) + ' of the ' + str(len(QualityControl)) + ' reads missing a reference correspond \
#the correct design.')

cor = np.array(cor)
absError = np.array(absError)
print(len(cor))
print(sum(absError == 0))

def targetseq(designseq):
    guide = designseq[20:40]
    id = 100 + designseq[100:].index(guide)
    target = designseq[(id-13):(id+52)]
    return target


_ = plt.hist(absError, bins='auto')  # arguments are passed to np.histogram
plt.title("Absolute Error without threshold [output - test]")
#plt.show()

p = plt.hist(cor, bins='auto')  # arguments are passed to np.histogram
plt.title("Correlation of output vs test")
#plt.show()
