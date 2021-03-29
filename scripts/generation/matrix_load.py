import pickle as pkl
from typing import Callable, Tuple

import numpy as np
from scipy import stats

from gen_mh_features import create_train_matrix

workdir = '../cwd/'


class LindelProfile:

    @property
    def guide(self):
        return self._guide

    @property
    def labels_repetition(self) -> np.ndarray:
        return self._labels

    @property
    def labels(self):
        labels_arr: np.ndarray = np.add(self._labels[0], self._labels[1])
        labels_arr += self._labels[2]
        return labels_arr

    @property
    def sequence(self):
        return self._sequence

    def __init__(self, sequence, guide, repetitions):
        self._sequence = sequence
        self._guide = guide
        self._labels = np.array([np.zeros(557, dtype='float32') for _ in repetitions])

    def reads(self):
        return np.sum(self.labels)

    def partitioned_reads(self):
        return np.array([sum(self.labels_repetition[x]) for x in range(0, len(self.labels_repetition))])

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
    def lindel_profiles(self):
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

    def filter(self, filterer: Callable[[LindelProfile], bool], key=False, on=None):
        if on is None:
            on = self.lindel_profiles.items()

        filter_profile: Callable[[Tuple[str, LindelProfile]], bool] = lambda item: filterer(
            item[0 if key is True else 1])
        self._profiles = dict(filter(filter_profile, on))
        return self

    def filter_corr_coef(self, corr_coef: float):
        for gd in self.lindel_profiles:
            profile = self.lindel_profiles[gd]
            rep_reads = profile.labels_repetition
            for idx in range(self._repetitions):
                corr, _ = stats.pearsonr(rep_reads[idx], rep_reads[(idx + 1) % 3])
                if round(corr, ndigits=2) < corr_coef:
                    rep_reads[idx] = np.zeros(557)

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
                diff = seq_valid[i] - self.lindel_profiles[seq_training[i]].labels
                mse = np.mean(diff ** 2)
                mse_dict[seq_training[i]] = mse
            else:
                missed += 1
        if missed > 0:
            print(f"Missed: {missed}")
        return mse_dict

    def write_profile(self, start_index=0, set_type='test', fraction=0.15):
        filename = self.work_dir + "Our_Lindel_{0}.txt"
        file = open(filename.format(set_type), 'w')
        size = round(len(self) * fraction)
        print(f"Writing {set_type} set of {size} entries")
        index = 0
        for profile in self.lindel_profiles.values():
            if index < start_index:
                index += 1
                continue
            elif index >= size:
                break

            train_matrix = create_train_matrix(profile.get_mh_input(), self.mh_features)
            file.write(profile.guide)
            file.write('\t')

            for feature in train_matrix[1:]:
                file.write(str(feature))
                file.write('\t')

            for label_idx, label in enumerate(profile.labels):
                file.write(str(label))
                if label_idx + 1 != len(profile.labels):
                    file.write('\t')

            if index + 1 != size:
                file.write("\n")
            index += 1
        file.close()

        return index

    def __len__(self):
        return len(self.lindel_profiles)


if __name__ == '__main__':
    # Construct profile generator
    generator = ProfileGenerator()

    # Generate insertions based on rep matrices
    generator.generate_insertions()

    gen_len = len(generator)
    print(f'Generated {gen_len} possible insertions')

    # Filter pearson .75 and reads > 10
    generator.filter_corr_coef(0.75)
    generator.filter(lambda item: item.reads() > 10)
    # Normalize
    generator.normalize_profiles()

    # Default writes test
    idx = generator.write_profile(fraction=.1)
    print(f'Wrote {idx} test entries')
    generator.write_profile(start_index=idx, set_type='training', fraction=.9)
    # from matplotlib import pyplot as plt
    #
    # x_range_label = np.arange(0, 1 ** -2, 1 ** -4)
    # x_range_label_str = list(map(lambda x: str(round(x, 1)), x_range_label))
    # x_range = (10 ** -10) * x_range_label
    # plt.hist(list(mses.values()), bins=round(len(mses) * 5), edgecolor='grey', alpha=0.4,
    #          label="MSE occurrence", dpi=144, figsize=(10, 5))
    # plt.xlim(0, 10 ** -5)
    # plt.xlabel("MSE between Generated and Precomputed'")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.title(f"Comparison of {len(mses.values())} profiles ({4790 - len(mses)} were not found)")
    # plt.show()
    #
    # mses_list = list(mses.values())
    # mses_list.sort()
    # outliers = mses_list[:-100]
    # mses_outliers = {(y, outliers.index(y)) if y in outliers else None for (x, y) in mses.items()}
    # filter(lambda x: x is not None, mses_outliers)
    # plt.scatter(mses_outliers[:, 0], mses_outliers[:, 1])
    # plt.xlabel("MSE")
    # plt.ylabel("100 outliers")
    # plt.show()
