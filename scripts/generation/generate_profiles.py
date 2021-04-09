import pickle as pkl
from typing import Dict, Callable, Any

import numpy as np

from scripts.generation.generate_mh_features import create_train_matrix


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

    def labels_together(self) -> np.ndarray:
        """
        Sums the label frequencies
        @return: All the different replicates summed
        """
        return self._labels.sum(axis=0)

    def reads(self):
        """
        The total amount of reads over all of the replicates
        @return:
        """
        return np.sum(self.labels_together())

    def partitioned_reads(self) -> np.ndarray:
        """
        Partitions the reads based on the replicate
        @return: an array containing the amount of reads each replicate has
        """
        return np.array([sum(self._labels[x]) for x in range(len(self._labels))])

    def count_from_rep(self, rep: int, index: int) -> None:
        """
        Adds a count of a class to a replicate in the labels matrix
        @param rep: The replicate from which the count came
        @param index: The index of the class
        """
        self._labels[rep, index] += 1

    def normalize(self) -> None:
        """
        Normalizes all the labels based on the total amount of reads.
        """
        reads = self.reads()
        for i in range(len(self._labels)):
            self._labels[i] /= reads

    def get_mh_input(self) -> str:
        """
        Gets the 65bp microhomology input to generate all the features
        @return: a string representing the 65 basepair sequence
        """
        index = self.sequence[100:].index(self.guide) + 100
        return self.sequence[index - 13:index + 47]

    @staticmethod
    def class_label(row: list) -> str:
        """
        Generates the class label of a row of a replicate matrix. Performs it as described in the paper.
        @param row: The row to generate the label for
        @return: The class label as a string which can be used to find the index of the class in the frequency array
        """
        if row[9] == 'del' and row[10] < 2 and row[11] < 30:
            return str(row[10]) + '+' + str(row[11])
        elif row[9] == 'ins' and row[11] < 3:
            return str(row[11]) + '+' + row[12]
        elif row[9] == 'ins' and row[11] > 2:
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
        self._repetition_matrices = self._load_replicates()
        self._label_map, self._features = self._load_feature_index()

    def _load_replicates(self) -> dict:
        """
        Loads all the replicates based on the range defined in the object and the replicate file list.
        Unpickles the file and adds it to the result.

        @return: The result dict containing the replicate mapping to the respective matrix
        """
        result = dict()
        for rep in self.repetition_range:
            file_name = self.repetition_files[rep]
            file = open(file_name, 'rb')
            rep_matrix = pkl.load(file)
            file.close()
            result[rep] = rep_matrix

        return result

    def _load_feature_index(self):
        """
        Opens the feature_index_all file to get the label to index mapping that is defined in this file.
        @return:
        """
        file = open(self._feature_index_file, 'rb')
        label, rev_index, features = pkl.load(file)
        file.close()
        return label, features

    def _init_profiles(self):
        """
        Uses the algient_NHEJ_guides_final file to initialize all the lindel profile instances for the 70k design
        @return: A dictionary of the profiles mapping the guide sequence to the profile instance
        """
        ref = open(self._algient_file)
        profiles = {}
        for line in ref:
            seq, label = line.rstrip('\r\n').split('\t')
            if '70k' in label:
                guide = seq[20:40]
                profile = LindelProfile(seq, guide, self.repetition_range)
                profiles[guide] = profile

        return profiles

    def generate_profiles(self) -> None:
        """
        Function to generate the profiles from the replicate matrices. Loops over every replicate matrix
        and then over every row in these matrices. For each row a class label is constructed and the
        labels_to_index dict is used to find at what index this row describes a read.

        We then get the guide/target from the matrix row and check if this guide exists in our lindel profiles dict
            (since there are reads in the matrices that are not present in the 70k design).

        Then we get the relevant profile from the dict and call the function count_from_rep using the curren rep value
        and the label index.
        """
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

    def _filter_(self, filterer: Callable[[dict, *Any], None], *args: Any):
        """
        Helper function to duplicate code writing. It takes a filter (filterer) as argument and calls that function
        using an empty dict and *args provided to this function that are specific to the filter type.

        The generators profiles are then updated by the dict that the filter function has altered to contain
        only the sequences to keep.

        @param filterer: The function that performs the filter
        @param args: any args the filterer requires
        @return: The amount of profiles dropped because of this filter
        """
        before = len(self)
        profiles_filtered = {}
        filterer(profiles_filtered, *args)
        self._profiles = profiles_filtered
        return before - len(self)

    def _filter_profiles_reads_(self, profiles_filtered: dict, reads: int):
        """
        For every sequence and profile belonging to this sequence the partitioned reads are computed. If any of
        these reads is bigger or equal than the passed reads parameter the profile is added to the filtered_profiles
        dict.

        @param profiles_filtered: The dict containing the sequence mapping to the profile that eventually will
            be used to update the internal profile representation
        @param reads: The reads threshoold at which a profile is added to this dict
        """
        for seq, profile in self.lindel_profiles.items():
            part_reads = profile.partitioned_reads()
            for rep in self.repetition_range:
                if part_reads[rep] >= reads:
                    profiles_filtered[seq] = profile
                    break

    def filter_profile_reads(self, reads: int) -> int:
        """
        A function that can be called to filter the profile reads. The real logic is contained within the
        _filter_profile_reads_ function, this simply chains the order of calls.

        @param reads: the minimum reads to be included as a profile
        @return: The result of _filter_ which will be the amount of profiles dropped because of the reads filter
        """
        return self._filter_(self._filter_profiles_reads_, reads)

    def _filter_profiles_pearson_(self, profiles_filtered, pearson: float):
        """
        Filters profiles based on their pearson correlation. It cyclically loops over the repetition range using the
        modular operator and checks the pearson correlation for every possible combination that way. The correlations
        are all stored in an array and we verify that the minimal correlation is greater than or equal to the passed
        pearson requirement.

        @param profiles_filtered: The dict containing the sequence mapping to the profile that eventually will
            be used to update the internal profile representation
        @param pearson: The pearson correlation below which profiles should be dropped.
        """
        for seq, profile in self.lindel_profiles.items():
            correlations = np.zeros(3)
            for idx in self.repetition_range:
                nxt = (idx + 1) % 3
                corr = np.corrcoef(profile.labels_repetition[idx], profile.labels_repetition[nxt])[1, 0]
                correlations[idx] = corr

            if len(correlations) > 0 and correlations[np.argmin(correlations)] >= pearson:
                profiles_filtered[seq] = profile

    def filter_profiles_pearson(self, pearson: float) -> int:
        """
        A function that wraps the pearson filter and calls the _filter_ chain to prevent code duplication. The logic
        for the actual filtering is contained within _filter_profiles_pearson_

        @param pearson: The minimal pearson threshold
        @return: The result of _filter_ which will be the amount of profiles dropped because of the perason filter
        """
        return self._filter_(self._filter_profiles_pearson_, pearson)

    def normalize_profiles(self) -> None:
        """
        Normalizes all of the profiles
        """
        for guide in self.lindel_profiles:
            profile = self.lindel_profiles[guide]
            profile.normalize()

    def mse(self, file: str) -> dict:
        """
        Computes a dict of mses between a given file containing profiles and this profile generator's profiles.

        @param file: The filename input, it is expected to be in the
            same directory as the workdir defined in the __init__
        @return: A dictionary mapping targets to their mse
        """
        training = np.loadtxt(f'{self.work_dir + file}.txt', delimiter="\t", dtype=str)
        seq_training = training[:, 0]
        # Dont care about features/seq only profile rn
        seq_valid = training[:, 3034:].astype('float32')
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

    def write_profile(self, amount, start_index: int = 0, set_type: str = 'test'):
        """
        Writes the profile to a set_type. Takes arguments such as start_index and amount so that splits can be
        created

        @param amount: How many entries to write to this set
        @param start_index: From what index the writing should start
        @param set_type: The name of the type of set we are writing to
        @return: The index that we ended at after writing the required amount of entries such that it can be used
            to write further sets starting at this index
        """
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

            # We tried writing profiles that took redundancy matrix into account but saw no difference
            # cmax = profile.get_redundancy_matrix(self.labels_to_index)
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
    """
    Constructs a ProfileGenerator and generates the profiles. It then drops based on reads and based on pearson
    correlation. Information about how many were dropped at each step is printed. All the profiles are normalized in the
    end and then the instance is returned.

    @param work_dir: The work_dir which the profile generator should be initialized with
    @return: The profile generator after all these operations
    """
    p_gen = ProfileGenerator(work_dir=work_dir)
    p_gen.generate_profiles()

    print(f'Found {len(p_gen)} possible sequences that correspond to a profile')

    # Filter pearson .75 and reads > 10
    read_drops = p_gen.filter_profile_reads(10)
    print(f'Left with {len(p_gen)} after {read_drops} < 10 reads drops')

    corr_drops = p_gen.filter_profiles_pearson(0.75)

    print(f'Left with {len(p_gen)} after {corr_drops} < .75 pearson correlation drops')
    # Normalize
    p_gen.normalize_profiles()
    return p_gen


if __name__ == '__main__':
    generator = construct_generator(work_dir='../../cwd/')
    # Default writes test
    test_size = 440
    idx_next = generator.write_profile(amount=test_size)
    generator.write_profile(amount=(len(generator) - test_size), start_index=idx_next, set_type='training')
