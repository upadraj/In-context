import numpy as np
import pandas as pd

from data_utils import task_to_keys


def _select_subset_by_ids(dataset, indices):  # indices is a list or np array here...
    subset = dataset.select(indices)
    return subset


def _select_subset_by_idx(dataset, indices):
    dataset = dataset.filter(lambda s: s["idx"] in indices)
    return dataset


def get_balanced_subsets(dataset):
    subset_per_label = {}
    for label_idx, _ in enumerate(dataset.features["label"].names):
        subset_per_label[label_idx] = dataset.filter(lambda s: s["label"] == label_idx)
    return subset_per_label


def _select_random_subset(dataset, num_shots, balanced=False, seed=123):
    # fix seed
    np.random.seed(seed)

    if num_shots < 1:
        return [], []

    if balanced:
        assert (
            num_shots % 2 == 0
        ), "a balanced context requires at least one demonstartion per label"
        # select the same number of samples from every label
        indices = []  # we collect all indices here
        subset_per_label = get_balanced_subsets(dataset)

        for _, samples in subset_per_label.items():
            subset_indices = samples["idx"]
            # select num_shots // 2 samples
            subset_indices = np.random.choice(
                subset_indices, size=num_shots // 2, replace=False
            )
            indices += list(subset_indices)
        assert len(indices) == num_shots
    else:
        # just select a random subset of samples
        indices = np.random.choice(dataset["idx"], size=num_shots, replace=False)

    # return _select_subset_by_ids(dataset, indices), indices
    return _select_subset_by_idx(dataset, indices), indices


def select_demonstrations(
    dataset,
    balanced=False,
    shuffle=False,
    from_indices=None,
    from_idxlabels=None,
    rand_subset=False,
    num_shots=16,
    seed=123,
):
    if from_indices is not None:
        demonstrations = _select_subset_by_ids(dataset, from_indices)
        indices = np.array(from_indices)
    elif from_idxlabels is not None:
        demonstrations = _select_subset_by_idx(dataset, from_idxlabels)
        indices = np.array(from_idxlabels)
    elif rand_subset:
        demonstrations, indices = _select_random_subset(
            dataset, num_shots, balanced, seed
        )
    else:
        demonstrations = dataset
        indices = np.array(dataset["idx"])

    if shuffle:
        if len(demonstrations) > 0:
            demonstrations = demonstrations.shuffle(seed + 1)

    return demonstrations, indices


def create_few_shot_context(
    dataset_name,
    demonstrations,
    int_to_label_converter,  # dataset.features["label"]
    teacher_description="",
    student_description="Are the following sentences examples of entailment, yes or no?",
    remove_label=False,
    pattern_name="default",
):
    # TODO: ? add support for more patterns
    separate_description_by = "\n\n"
    separate_shots_by = "\n\n"

    # create context
    context = (
        ""
        if teacher_description == ""
        else f"{teacher_description}{separate_description_by}"
    )
    student_context = (
        ""
        if student_description == ""
        else f"{student_description}{separate_description_by}"
    )

    if pattern_name == "explanation":
        pattern = "{prefix1}: {text1}\n{prefix2}: {text2}\nExplanation: {explanation}"
    else:
        pattern = "{prefix1}: {text1}\n{prefix2}: {text2}"

    current_shot = len(demonstrations)
    for sample in demonstrations:
        formated_sample = pattern.format(
            prefix1=task_to_keys[dataset_name][0].capitalize(),
            text1=sample[task_to_keys[dataset_name][0]],
            prefix2=task_to_keys[dataset_name][1].capitalize(),
            text2=sample[task_to_keys[dataset_name][1]],
            explanation=(
                sample["explanation"] if "explanation" in sample.keys() else None
            ),
        )
        if sample["label"] == -1 or remove_label:
            verbalized_label = ""
        elif current_shot == 1:
            verbalized_label = ""
            student_context += (
                f"{formated_sample}\nLabel: {verbalized_label}{separate_shots_by}"
            )
        else:
            verbalized_label = int_to_label_converter.int2str(sample["label"])
        context += f"{formated_sample}\nLabel: {verbalized_label}{separate_shots_by}"
        current_shot -= 1

    context = context.strip()
    student_context = student_context.strip()
    return context, student_context
