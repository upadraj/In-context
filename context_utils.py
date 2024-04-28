import numpy as np
import pandas as pd

from datasets import concatenate_datasets

from data_utils import task_to_keys


def _select_subset_by_ids(dataset, indices):  # indices is a list or np array here...
    subset = dataset.select(indices)
    return subset


def _select_subset_by_idx(dataset, indices):
    dataset = dataset.filter(lambda s: s["idx"] in indices)
    return dataset


def _filter_subset_by_idx(dataset, indices):
    dataset = dataset.filter(lambda s: s["idx"] not in indices)
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
    return _select_subset_by_idx(dataset, indices)


def select_demonstrations(
    dataset,
    balanced=False,
    shuffle=False,
    from_indices=None,
    from_idxlabels=None,
    rand_subset=False,
    num_shots=16,
    seed=123,
    filter_out_idx=None,
):
    if filter_out_idx:
        dataset = _filter_subset_by_idx(dataset, filter_out_idx)
    if from_indices is not None:
        demonstrations = _select_subset_by_ids(dataset, from_indices)
    elif from_idxlabels is not None:
        demonstrations = _select_subset_by_idx(dataset, from_idxlabels)
    elif rand_subset:
        demonstrations = _select_random_subset(dataset, num_shots, balanced, seed)
    else:
        demonstrations = dataset

    if shuffle:
        if len(demonstrations) > 0:
            demonstrations = demonstrations.shuffle(seed + 1)

    indices = np.array(demonstrations["idx"])
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
        if dataset_name == 'paws-qqp':
            formated_sample = pattern.format(
                prefix1="Question1",
                text1=sample[task_to_keys[dataset_name][0]],
                prefix2="Question2",
                text2=sample[task_to_keys[dataset_name][1]],
                explanation=(
                    sample["explanation"] if "explanation" in sample.keys() else None
                ),
            )
        else:
            formated_sample = pattern.format(
                prefix1=task_to_keys[dataset_name][0].capitalize(),
                text1=sample[task_to_keys[dataset_name][0]],
                prefix2=task_to_keys[dataset_name][1].capitalize(),
                text2=sample[task_to_keys[dataset_name][1]],
                explanation=(
                    sample["explanation"] if "explanation" in sample.keys() else None
                ),
            )
        student_label = ""
        if sample["label"] == -1 or remove_label:
            verbalized_label = ""
        elif current_shot == 1:
            verbalized_label = ""
            student_context += (
                f"{formated_sample}\nLabel: {verbalized_label}{separate_shots_by}"
            )
            student_label = int_to_label_converter.int2str(sample["label"])
        else:
            verbalized_label = int_to_label_converter.int2str(sample["label"])
        context += f"{formated_sample}\nLabel: {verbalized_label}{separate_shots_by}"
        current_shot -= 1

    context = context.strip()
    student_context = student_context.strip()
    return context, student_context, student_label


def create_train_batch_token(
    dataset_name,
    datasets,
    tokenizer,
    seed,
    teacher_description="",
    student_description="Are the following sentences examples of entailment, yes or no?",
    num_shots=16,
    device="cpu",
    num_train_samps=128,
):
    datasets = datasets["train"]

    batch_tokens = []
    batch_strings = []
    all_indices = []
    student_labels = []

    contexts, context_indices = select_demonstrations(
        datasets, balanced=True, rand_subset=True, num_shots=num_shots, seed=seed
    )
    candidates, candidate_indices = select_demonstrations(
        datasets, filter_out_idx=context_indices.tolist()
    )
    candidates_chosen = np.random.choice(
        candidate_indices, num_train_samps, replace=False
    ).tolist()
    chosen_candidates, chosen_indices = select_demonstrations(
        datasets, from_idxlabels=candidates_chosen
    )

    for samp in candidates_chosen:
        demonstrations = chosen_candidates.filter(lambda r: r["idx"] == samp)
        complete_set = concatenate_datasets([contexts, demonstrations])
        context, student_context, student_label = create_few_shot_context(
            dataset_name,
            complete_set,
            complete_set.features["label"],
            teacher_description=teacher_description,
            student_description=student_description,
        )
        token_data = {
            "context": (tokenizer(context, return_tensors="pt")).to(device),
            "query": (tokenizer(student_context, return_tensors="pt")).to(device),
        }
        string_data = {"context": context, "query": student_context}
        batch_tokens.append(token_data)
        batch_strings.append(string_data)
        all_indices.append(samp)
        student_labels.append(student_label)
    return batch_tokens, batch_strings, all_indices, context_indices, student_labels


def create_validation_batch_token(
    dataset_name,
    datasets,
    tokenizer,
    device="cpu",
    prompt_descr="Are the following sentences examples of entailment, yes or no?",
    limit=128,
    shuffle=False,
    seed=42
):
    if dataset_name == "mnli":
        split = "validation_matched"
    else:
        split = "validation"

    datasets = datasets[split]

    if shuffle:
        demonstrations, all_indices = select_demonstrations(datasets, shuffle=shuffle,seed=seed)
    else:
        demonstrations, all_indices = select_demonstrations(datasets)
    batch_tokens = []
    batch_strings = []
    for dx in range(limit):
        context, _, _ = create_few_shot_context(
            dataset_name,
            [demonstrations[dx]],
            demonstrations.features["label"],
            teacher_description=prompt_descr,
            remove_label=True,
        )
        token_data = (tokenizer(context, return_tensors="pt")).to(device)
        batch_tokens.append(token_data)
        batch_strings.append(context)

    labels = [datasets["label"][datasets["idx"].index(i)] for i in all_indices]
    labels = datasets.features["label"].int2str(labels)
    return batch_tokens, batch_strings, all_indices[:limit], labels[:limit]


def create_paws_qqp_batch_token(
    datasets,
    tokenizer,
    device="cpu",
    prompt_descr="Are the following sentences examples of entailment, yes or no?",
    limit=128,
    shuffle=False,
    seed=42
):
    if shuffle:
        demonstrations, all_indices = select_demonstrations(datasets, shuffle=shuffle,seed=seed)
    else:
        demonstrations, all_indices = select_demonstrations(datasets)
    batch_tokens = []
    batch_strings = []
    for dx in range(limit):
        context, _, _ = create_few_shot_context(
            "paws-qqp",
            [demonstrations[dx]],
            demonstrations.features["label"],
            teacher_description=prompt_descr,
            remove_label=True,
        )
        token_data = (tokenizer(context, return_tensors="pt")).to(device)
        batch_tokens.append(token_data)
        batch_strings.append(context)

    labels = [datasets["label"][datasets["idx"].index(i)] for i in all_indices]
    labels = datasets.features["label"].int2str(labels)
    return batch_tokens, batch_strings, all_indices[:limit], labels[:limit]


def create_hans_batch_token(
    datasets,
    tokenizer,
    device="cpu",
    prompt_descr="Are the following sentences examples of entailment, yes or no?",
    limit=128,
    shuffle=True,
    seed=42
):
    if shuffle:
        demonstrations, all_indices = select_demonstrations(datasets, shuffle=shuffle,seed=seed)
    else:
        demonstrations, all_indices = select_demonstrations(datasets)
    batch_tokens = []
    batch_strings = []
    for dx in range(limit):
        context, _, _ = create_few_shot_context(
            "hans",
            [demonstrations[dx]],
            demonstrations.features["label"],
            teacher_description=prompt_descr,
            remove_label=True,
        )
        token_data = (tokenizer(context, return_tensors="pt")).to(device)
        batch_tokens.append(token_data)
        batch_strings.append(context)

    labels = [datasets["label"][datasets["idx"].index(i)] for i in all_indices]
    labels = datasets.features["label"].int2str(labels)
    return batch_tokens, batch_strings, all_indices[:limit], labels[:limit]
