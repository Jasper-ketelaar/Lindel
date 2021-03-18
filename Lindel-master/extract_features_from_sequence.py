import numpy as np

def label_mh(sample,mh_len):
    '''Function to label microhomology in deletion events'''
    for k in range(len(sample)):
        read = sample[k]
        # We are going to check for all deletions whether they contain an MH
        if read[3] == 'del':
            idx = read[2] + read[4] +17 # read[2] is always 13 --> idx = start position of the deletion
            idx2 = idx + read[5] # = position of the last deleted bp
            x = mh_len if read[5] > mh_len else read[5] # length of MH is max(mh_len,del_len)
            # check for MH of max length x
            for i in range(x,0,-1):
                # MH: seq upstream of del start = seq upstream of del end
                if read[1][idx-i:idx] == read[1][idx2-i:idx2] and i <= read[5]:
                    sample[k][-2] = 'mh'
                    sample[k][-1] = i
                    break
            if sample[k][-2]!='mh':
                sample[k][-1]=0
    return sample

def gen_indel(sequence,cut_site):
    '''This is the function that used to generate all possible unique indels and
    list the redundant classes which will be combined after'''
    nt = ['A','T','C','G']
    up = sequence[0:cut_site]
    down = sequence[cut_site:]
    dmax = min(len(up),len(down))
    uniqe_seq ={}
    # Generate all possible deletions
    # Start
    for dstart in range(1,cut_site+3):
        for dlen in range(1,dmax):
            if len(sequence) > dlen+dstart > cut_site-2:
                seq = sequence[0:dstart]+sequence[dstart+dlen:]
                indel = sequence[0:dstart] + '-'*dlen + sequence[dstart+dlen:]
                array = [indel,sequence,13,'del',dstart-30,dlen,None,None,None]
                try:
                    uniqe_seq[seq]
                    if dstart-30 <1:
                        uniqe_seq[seq] = array
                except KeyError: uniqe_seq[seq] = array
    # Generate all possible Insertion (1-2 bp)
    for base in nt:
        seq = sequence[0:cut_site]+base+sequence[cut_site:]
        indel = sequence[0:cut_site]+'-'+sequence[cut_site:]
        array = [sequence,indel,13,'ins',0,1,base,None,None]
        try: uniqe_seq[seq] = array
        except KeyError: uniqe_seq[seq] = array
        for base2 in nt:
            seq = sequence[0:cut_site] + base + base2 + sequence[cut_site:]
            indel = sequence[0:cut_site]+'--'+sequence[cut_site:]
            array = [sequence,indel,13,'ins',0,2,base+base2,None,None]
            try: uniqe_seq[seq] = array
            except KeyError:uniqe_seq[seq] = array
    # add MH labels to deletion events containing an MH
    uniq_align = label_mh(list(uniqe_seq.values()),4)
    # Redundant/Indistinguishable classes of MH-deletions
    for read in uniq_align:
        if read[-2]=='mh':
            merged=[]
            for i in range(0,read[-1]+1):
                merged.append((read[4]-i,read[5]))
            read[-3] = merged
    return uniq_align

def create_feature_array(ft,uniq_indels):
    '''Used to create microhomology feature array
       require the features and label
    '''
    ft_array = np.zeros(len(ft))
    for read in uniq_indels:
        if read[-2] == 'mh':
            # MH label: [start position + deletion length + MH length]
            mh = str(read[4]) + '+' + str(read[5]) + '+' + str(read[-1])
            try:
                ft_array[ft[mh]] = 1
            except KeyError:
                pass
        else:
            pt = str(read[4]) + '+' + str(read[5]) + '+' + str(0)
            try:
                ft_array[ft[pt]]=1
            except KeyError:
                pass
    return ft_array


def onehotencoder(seq):
    '''convert to single and di-nucleotide hotencode'''
    nt= ['A','T','C','G']
    head = []
    l = len(seq)

    # list all Single-Nucleotide seq-pos options ['N#'] (length = 4*L)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))
    # list all Di-Nucleotide seq-pos options ['NN#] (length = 16*(L-1))
    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    # Dict: seq-label --> 1-hot index
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    # set applicable single and dinucleotide features to 1
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode

def create_train_matrix(seq,features):
    # create a train / test matrix including the features.
    # seq = 65 bp sequence with a PAM starting at position 33
    # features = mh features from prereq
    ind = gen_indel(seq,30)
    guide = seq[13:33]
    mh_features = create_feature_array(features,ind)
    sequence_features = onehotencoder(guide)
    res = np.concatenate((seq,mh_features,sequence_features),axis=None)
    return res

