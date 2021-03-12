import pickle as pkl

import numpy as np

if __name__ == '__main__':
    wdir = '../cwd/'
    ref_table = {}
    ref = open(f'{wdir}algient_NHEJ_guides_final.txt')
    for line in ref:
        seq, label = line.rstrip('\r\n').split('\t')
        guide = seq[20:40]
        if guide not in ref_table:
            ref_table[guide] = [label]

    matrix_iterations = []
    for i in range(1, 4):
        rep_file = open(f'{wdir}NHEJ_rep{i}_final_matrix.pkl', 'rb')
        rep_matrix = pkl.load(rep_file)
        rep_file.close()
        matrix_iterations.append(rep_matrix)

    matrix_file = open(f'{wdir}NHEJ_MH_final_matrix.pkl', 'rb')
    mh_matrix = pkl.load(matrix_file)
    matrix_file.close()
    matrix_iterations.append(mh_matrix)

    features_file = open(f'{wdir}feature_index_all.pkl', 'rb')
    label, rev_index, features = pkl.load(features_file)
    features_file.close()
    print(label)
    freq = {}
    for matrix in matrix_iterations:

        for mh in matrix:
            alg = ref_table[mh[6]]
            indel = mh[9]
            target = mh[6]

            if indel == 'del':
                cut_site = mh[10] + mh[4]
                print(mh)
                mh_label = f'{"-" if mh[11] != 0 else ""}{mh[11]}+{cut_site}'
                print(mh_label)
                if mh_label in label:
                    if target not in freq:
                        freq[target] = {}
                    mh_index = label[mh_label]
                    if mh_index in freq[target]:
                        freq[target][mh_index] += 1
                    else:
                        freq[target][mh_index] = 1

            elif indel == 'ins':
                ins_len = mh[-4]
                ins_label = ''
                if ins_len > 2:
                    ins_label = '3'
                else:
                    ins_bp = mh[-3]
                    if ins_bp == 'N':
                        continue
                    ins_label = f'{ins_len}+{ins_bp}'
                if target not in freq:
                    freq[target] = {}
                label_idx = label[ins_label]
                if label_idx in freq[target]:
                    freq[target][label_idx] += 1
                else:
                    freq[target][label_idx] = 1

    # Compare to their train set see if frequencies match at all
    training = np.loadtxt(f'{wdir}Lindel_training.txt', delimiter="\t", dtype=str)
    seq_training = training[:, 0]

    # Dont care about features/seq only profile rn
    seq_valid = training[:, 1 + 3033:]
    for freq_entry in freq:
        freq_sum = sum(freq[freq_entry].values())
        for freq_entry_label in freq[freq_entry]:
            freq[freq_entry][freq_entry_label] /= freq_sum

    for index, seq_sample in enumerate(seq_training):
        if seq_sample in freq:
            vals = seq_valid[index]
            for freq_idx in freq[seq_sample]:
                print(f'{vals[freq_idx]}: {freq[seq_sample][freq_idx]}')
