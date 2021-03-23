import pickle as pkl
from collections import defaultdict

workdir = '../cwd/'


class LindelProfile:

    @property
    def guide(self):
        return self._guide

    @property
    def labels(self):
        return self._labels

    @property
    def sequence(self):
        return self._sequence

    def __init__(self, sequence, guide):
        self._sequence = sequence
        self._guide = guide
        self._labels = defaultdict(int)

    def __setitem__(self, label, index):
        self._labels[label] = index

    def __getitem__(self, item):
        return self.labels[item]

    def __iadd__(self, other):
        self._labels[other] += 1

    def sum(self):
        return sum(self.labels.values())


class ProfileGenerator:

    def __init__(
            self,
            work_dir="../../cwd/",
            algient_file="algient_NHEJ_guides_final.txt",
            rep_file="NHEJ_rep{0}_final_matrix.pkl",
            repetitions=3,
            feature_index_file="feature_index_all.pkl"
    ):
        self._work_dir = work_dir
        self._algient_file = work_dir + algient_file
        self._repetition_files = [work_dir + rep_file.format(x + 1) for x in range(repetitions)]
        self._feature_index_file = work_dir + feature_index_file
        self._repetition_matrices = self._load_repetitions()
        self._label_map = self._load_feature_index()
        self._frequencies = self._init_frequencies()

    def _load_repetitions(self):
        result = dict()
        for rep, file_name in enumerate(self._repetition_files):
            file = open(file_name, 'rb')
            rep_matrix = pkl.load(file)
            file.close()
            result[rep] = rep_matrix
        return result

    def _load_feature_index(self):
        file = open(self._feature_index_file, 'rb')
        label, _, _ = pkl.load(file)
        file.close()
        return label

    def _init_frequencies(self):
        ref = open(self._algient_file)
        profiles = {}
        for line in ref:
            seq, _ = line.rstrip('\r\n').split('\t')
            profile = LindelProfile(seq, seq[20:40])
            profiles[profile.guide] = profile

        return [profiles for _ in range(len(self._repetition_matrices))]

    def rep_matrix_row(self, rep, row):
        return self._repetition_matrices[rep][row]

    def indel_class_label(self, row=0, rep=0, data=None):
        if data is None:
            data = self.rep_matrix_row(rep, row)

        if data[9] == 'del' and data[10] < 2 and data[11] < 30:
            return str(data[10]) + '+' + str(data[11])
        elif data[9] == 'ins' and data[11] < 3:
            return str(data[11]) + '+' + data[12]
        elif data[9] == 'ins' and data[11] > 2:
            return '3'
        return ''

    def generate_insertions(self):
        for matrix in self._repetition_matrices:
            for row, column in enumerate(self._repetition_matrices[matrix]):
                label = self.indel_class_label(data=column)
                guide = column[6]
                design = column[5]
                try:
                    index = self._label_map[label]
                    if design == 'wt':
                        matrix_freqs = self._frequencies[matrix]
                        if guide in matrix_freqs:
                            profile = matrix_freqs[guide]
                            if profile is None:
                                continue
                            profile += index
                except KeyError:
                    pass


if __name__ == '__main__':
    generator = ProfileGenerator()
    generator.generate_insertions()
    # Loading final matrix data
    # rep1 = pkl.load(open(workdir + 'NHEJ_rep1_final_matrix.pkl', 'rb'))
    # rep2 = pkl.load(open(workdir + 'NHEJ_rep2_final_matrix.pkl', 'rb'))
    # rep3 = pkl.load(open(workdir + 'NHEJ_rep3_final_matrix.pkl', 'rb'))
    # # mh = pkl.load(open(workdir + 'NHEJ_MH_final_matrix.pkl', 'rb'))
    # # combined = np.vstack([rep1, rep2, rep3])
    # f2id = pkl.load(open(workdir + 'feature_index_all.pkl', 'rb'))
    #
    # # Dictionaries for missing features and missing 200mers
    # missingClasses = {}
    # missingSequence = {}
    #
    # # Dictionaries for data
    # r1 = indelevents(rep1)
    # r2 = indelevents(rep2)
    # r3 = indelevents(rep3)
    #
    # # Normalize events to generate probability density for indel classes
    # # Remove entries with an insufficient number of reads
    # th = 10
    # r1 = normalize(threshold(r1, th))
    # r2 = normalize(threshold(r2, th))
    # r3 = normalize(threshold(r3, th))
