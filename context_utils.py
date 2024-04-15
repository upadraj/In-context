import numpy as np
import pandas as pd

from data_utils import task_to_keys

# --pattern "{text1} {text2} ?" \
# --target_prefix " " \
# --target_tokens "ĠYes,ĠNo" \
# --separate_shots_by "\n\n" \

import numpy as np
import pandas as pd

from data_utils import task_to_keys

def get_sample_ids(path):
    # we save examples as a .csv file which has an "idx" column
    df = pd.read_csv(path, sep=",", index_col=0)
    return df["idx"].values

def _select_subset_by_ids(dataset, indices):
    subset = dataset.select(indices)
    return subset

def _select_subset_by_idx(dataset, indices):
    dataset = dataset.filter(
        lambda s: s["idx"] in indices)
    return dataset

def get_balanced_subsets(dataset):
    subset_per_label = {}
    for label_idx, _ in enumerate(dataset.features["label"].names):
        subset_per_label[label_idx] = dataset.filter(
            lambda s: s["label"] == label_idx)
    return subset_per_label

def _select_random_subset(dataset, num_shots, balanced=False, seed=123):
    # fix seed
    np.random.seed(seed)

    if num_shots < 1:
        return [], []

    if balanced:
        assert num_shots % 2 == 0, "a balanced context requires at least one demonstartion per label"
        # select the same number of samples from every label
        indices = []  # we collect all indices here
        subset_per_label = get_balanced_subsets(dataset)

        for _, samples in subset_per_label.items():
            subset_indices = samples["idx"]
            # select num_shots // 2 samples
            subset_indices = np.random.choice(
                subset_indices, size=num_shots // 2, replace=False)
            indices += list(subset_indices)
        assert len(indices) == num_shots
    else:
        # just select a random subset of samples
        indices = np.random.choice(
            dataset['idx'], size=num_shots, replace=False)

    # return _select_subset_by_ids(dataset, indices), indices
    return _select_subset_by_idx(dataset, indices), indices


def create_few_shot_context(
    dataset_name,
    dataset,
    num_shots,
    description="",
    remove_label=False,
    from_indices=None,
    balanced=False,
    shuffle=False,
    seed=123
):
    separate_description_by="\n\n"
    separate_shots_by="\n\n"
    # select samples from which the context will be constructed
    if from_indices is not None:
        demonstrations = _select_subset_by_ids(dataset, from_indices)
        indices = np.array(from_indices)
    else:
        demonstrations, indices = _select_random_subset(
            dataset, num_shots, balanced, seed)

    if shuffle:
        if len(demonstrations) > 0:
            demonstrations = demonstrations.shuffle(seed)

    # create context
    context = "" if description == "" else f"{description}{separate_description_by}"
    int_to_label_converter = dataset.features['label']

    if task_to_keys[dataset_name][1] is not None:
        pattern = '{prefix1}: {text1}\n{prefix2}: {text2}'
    else:
        pattern = '{prefix1}: {text1}'
    
    for sample in demonstrations:
        
        second_key_present = task_to_keys[dataset_name][1]
        formated_sample = pattern.format(
            prefix1=task_to_keys[dataset_name][0].capitalize(),
            text1=sample[task_to_keys[dataset_name][0]],
            prefix2=task_to_keys[dataset_name][1].capitalize() if second_key_present is not None else None,
            text2=sample[task_to_keys[dataset_name][1]] if second_key_present is not None else None
        )
        if sample["label"] == -1 or remove_label:
            verbalized_label = ""
        else:
            verbalized_label = int_to_label_converter.int2str(sample["label"])
        context += f"{formated_sample}\nLabel: {verbalized_label}{separate_shots_by}"

    return context, indices