import pickle as pkl
from typing import Dict

import numpy as np

from scripts.generation.gen_mh_features import create_train_matrix


class LindelProfile:

    @property
    def guide(self):
        return self._guide

    @property
    def labels_repetition(self):
        return self._labels

    @property
    def sequence(self):
        return self._sequence

    def __init__(self, sequence, guide, repetitions):
        self._sequence = sequence
        self._guide = guide
        self._labels = np.array([np.zeros(557, dtype='float32') for _ in repetitions])

    def labels_together(self):
        result = np.zeros(557)
        for idx in range(0, len(self._labels)):
            if self.is_rep_valid(idx):
                result += self._labels[idx]
        return result

    def reads(self):
        return np.sum(self.labels_together())

    def is_valid(self):
        return self.is_rep_valid((1, 2)) or self.is_rep_valid((0, 1)) or self.is_rep_valid((0, 2))

    def labels_valid(self) -> np.ndarray:
        return np.array(list(filter(lambda x: self._labels[x] is not None, self._labels)))

    def partitioned_reads(self) -> np.ndarray:
        return np.array([sum(self._labels[x]) if self.is_rep_valid(x) else 0 for x in range(0, len(self._labels))])

    def is_rep_valid(self, rep):
        start = rep
        end = rep
        if isinstance(rep, tuple):
            start = rep[0]
            end = rep[1]
        return not np.isnan(np.dot(self._labels[start], self._labels[end]))

    def count_from_rep(self, rep: int, index: int):
        self._labels[rep, index] += 1

    def average_rep_reads(self):
        return self.partitioned_reads().mean()

    def normalize(self):
        reads = self.reads()
        for i in range(len(self._labels)):
            self._labels[i] /= reads

    def get_mh_input(self):
        index = self.sequence[100:].index(self.guide) + 100
        return self.sequence[index - 13:index + 52]

    def aggregate_guess(self):
        return np.argmax(self.labels_together())

    def drop_repetition(self, repetition):
        self._labels[repetition] = np.newaxis

    @staticmethod
    def class_label(data):
        if data[9] == 'del' and data[10] < 2 and data[11] < 30:
            return str(data[10]) + '+' + str(data[11])
        elif data[9] == 'ins' and data[11] < 3:
            return str(data[11]) + '+' + data[12]
        elif data[9] == 'ins' and data[11] > 2:
            return '3'

        return ''


class ProfileGenerator:

    @property
    def work_dir(self) -> str:
        return self._work_dir

    @property
    def repetition_range(self) -> range:
        return range(0, self._repetitions)

    @property
    def algient_file_name(self):
        return self._algient_file

    @property
    def repetition_files(self):
        return [self._rep_file.format(x + 1) for x in self.repetition_range]

    @property
    def repetition_matrices(self):
        return self._repetition_matrices

    @property
    def lindel_profiles(self) -> Dict[str, LindelProfile]:
        return self._profiles

    @property
    def labels_to_index(self):
        return self._label_map

    @property
    def mh_features(self):
        return self._features

    def __init__(
            self,
            work_dir="../../cwd/",
            algient_file="algient_NHEJ_guides_final.txt",
            rep_file="NHEJ_rep{0}_final_matrix.pkl",
            repetitions=3,
            feature_index_file="feature_index_all.pkl"
    ):
        self._work_dir = work_dir
        self._repetitions = repetitions
        self._algient_file = work_dir + algient_file
        self._rep_file = work_dir + rep_file
        self._feature_index_file = work_dir + feature_index_file
        self._profiles = self._init_profiles()
        self._repetition_matrices = self._load_repetitions()
        self._label_map, self._features = self._load_feature_index()

    def _load_repetitions(self):
        result = dict()
        for rep in self.repetition_range:
            file_name = self.repetition_files[rep]
            file = open(file_name, 'rb')
            rep_matrix = pkl.load(file)
            file.close()
            result[rep] = rep_matrix

        return result

    def _load_feature_index(self):
        file = open(self._feature_index_file, 'rb')
        label, rev_index, features = pkl.load(file)
        file.close()
        return label, features

    def _init_profiles(self):
        ref = open(self._algient_file)
        profiles = {}
        for line in ref:
            seq, label = line.rstrip('\r\n').split('\t')
            if '70k' in label:
                guide = seq[20:40]
                profile = LindelProfile(seq, guide, self.repetition_range)
                profiles[guide] = profile

        return profiles

    def rep_matrix_row(self, rep, row):
        return self._repetition_matrices[rep][row]

    def generate_insertions(self):
        for rep in self.repetition_range:
            matrix = self.repetition_matrices[rep]
            for obs in matrix:
                label = LindelProfile.class_label(obs)
                if label not in self.labels_to_index:
                    continue
                guide = obs[6]
                if guide not in self.lindel_profiles:
                    continue

                label_index = self.labels_to_index[label]

                profile = self.lindel_profiles[guide]
                profile.count_from_rep(rep, label_index)

    def filter_profiles(self, reads):
        for seq in self.lindel_profiles:
            profile = self.lindel_profiles[seq]
            part_reads = profile.partitioned_reads()
            drops = 0
            for rep in self.repetition_range:
                if part_reads[rep] < reads:
                    profile.drop_repetition(rep)
                    drops += 1

    def filter_corr_coef(self, corr_coef: float):
        for gd in self.lindel_profiles:
            profile = self.lindel_profiles[gd]
            for idx in self.repetition_range:
                nxt = (idx + 1) % 3
                if not profile.is_rep_valid((idx, nxt)):
                    continue
                corr = np.corrcoef(profile.labels_repetition[idx], profile.labels_repetition[nxt])[1, 0]
                if corr < corr_coef:
                    profile.drop_repetition(idx)

    def normalize_profiles(self):
        for guide in self.lindel_profiles:
            profile = self.lindel_profiles[guide]
            profile.normalize()

    def mse(self, file: str):
        training = np.loadtxt(f'{self.work_dir + file}.txt', delimiter="\t", dtype=str)
        seq_training = training[:, 0]
        # Dont care about features/seq only profile rn
        seq_valid = training[:, 1 + 3033:].astype('float32')
        mse_dict = dict()
        missed = 0
        for i in range(0, len(training)):
            if seq_training[i] in self.lindel_profiles:
                diff = seq_valid[i] - self.lindel_profiles[seq_training[i]].labels_together()
                mse = np.mean(diff ** 2)
                mse_dict[seq_training[i]] = mse
            else:
                missed += 1
        if missed > 0:
            print(f"Missed: {missed}")
        return mse_dict

    def write_profile(self, amount, start_index=0, set_type='test'):
        filename = self.work_dir + "Our_Lindel_{0}.txt"
        file = open(filename.format(set_type), 'w')
        print(f"Writing {set_type} set of {amount} entries")
        stop = (amount + start_index)
        index = 0
        for profile in self.lindel_profiles.values():
            if index < start_index:
                index += 1
                continue
            elif index == stop:
                break

            train_matrix = create_train_matrix(profile.get_mh_input(), self.mh_features)
            file.write(profile.guide)
            file.write('\t')

            for feature in train_matrix[1:]:
                file.write(str(feature))
                file.write('\t')

            profile_frequencies = profile.labels_together()
            profile_len = len(profile_frequencies)

            for label_idx, label_freq in enumerate(profile_frequencies):
                file.write(str(label_freq))
                if label_idx + 1 != profile_len:
                    file.write('\t')

            if index + 1 != stop:
                file.write("\n")
            index += 1
        file.close()

        return index

    def __len__(self):
        return len(self.lindel_profiles)


def construct_generator(work_dir='../cwd/'):
    p_gen = ProfileGenerator(work_dir=work_dir)

    # Generate insertions based on rep matrices
    p_gen.generate_insertions()

    gen_len = len(p_gen)
    print(f'Found {gen_len} possible insertions')

    # Filter pearson .75 and reads > 10
    p_gen.filter_profiles(10)
    count = 0
    for seq in p_gen.lindel_profiles:
        if p_gen.lindel_profiles[seq].is_valid():
            count += 1
    print(f'Left with {count} after reads drops')

    p_gen.filter_corr_coef(0.75)

    for seq, profile in p_gen.lindel_profiles.copy().items():
        if not profile.is_valid():
            p_gen.lindel_profiles.pop(seq)

    print(f'After filter we are left with {len(p_gen)} indels')
    # Normalize
    p_gen.normalize_profiles()
    return p_gen


if __name__ == '__main__':
    generator = construct_generator(work_dir='../../cwd/')
    # Default writes test
    test_size = 440
    idx_next = generator.write_profile(amount=test_size)
    generator.write_profile(amount=(len(generator) - test_size), start_index=idx_next, set_type='training')
