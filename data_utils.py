import pandas as pd
import numpy as np
from datasets import load_dataset, ClassLabel

task_to_keys = {
    # labels are: 0 (entailment), 1 (contradiction)
    "rte": ("sentence1", "sentence2"),  # labels are: 0 (entailment), 1 (not_entailment)
    "mnli": ("premise", "hypothesis"),
    "hans": ("premise", "hypothesis"),
    # labels are: 0 (not_duplicate), 1 (duplicate)
    "qqp": ("question1", "question2"),
    "paws-qqp": ("sentence1", "sentence2"),
}


def load_glue_datasets(task_name):
    # Get the datasets: specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if task_name is not None:
        if task_name == "mnli":
            # convert to binary format (remove neutral class)
            raw_datasets = load_dataset("glue", task_name)

            raw_datasets = raw_datasets.filter(lambda example: example["label"] != 1)

            # change labels of contradiction examples from 2 to 1
            def change_label(example):
                example["label"] = 1 if example["label"] == 2 else example["label"]
                return example

            raw_datasets = raw_datasets.map(change_label)

            # change features to reflect the new labels
            features = raw_datasets["train"].features.copy()
            features["label"] = ClassLabel(num_classes=2, names=["yes", "no"], id=None)
            raw_datasets = raw_datasets.cast(features)  # overwrite old features

        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset("glue", task_name)

            # change features to reflect the new labels
            features = raw_datasets["train"].features.copy()
            features["label"] = ClassLabel(num_classes=2, names=["yes", "no"], id=None)
            raw_datasets = raw_datasets.cast(features)  # overwrite old features

            if task_name == "qqp":
                # we subsample qqp already here because its really big
                # make sure we fix the seed here
                np.random.seed(123)
                for split in raw_datasets.keys():
                    raw_datasets[split] = raw_datasets[split].select(
                        np.random.choice(
                            np.arange(len(raw_datasets[split])),
                            size=1000,
                            replace=False,
                        )
                    )

    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    return raw_datasets, label_list, num_labels


def load_hans_dataset(heuristic="lexical_overlap"):
    # heuristic = {lexical_overlap, subsequence, constituent}
    # label = possible values including entailment (0), non-entailment (1)

    subset = "hans"
    raw_datasets = load_dataset("hans", split="validation")

    # hans comes without indices, so we add them
    indices = list(range(len(raw_datasets)))
    raw_datasets = raw_datasets.add_column(name="idx", column=indices)

    raw_datasets = raw_datasets.filter(
        lambda example: example["heuristic"] == heuristic
    )

    # change features to reflect the new labels
    features = raw_datasets.features.copy()
    features["label"] = ClassLabel(num_classes=2, names=["yes", "no"], id=None)
    raw_datasets = raw_datasets.cast(features)  # overwrite old features

    label_list = raw_datasets.features["label"].names
    num_labels = len(label_list)

    return raw_datasets, label_list, num_labels

def load_paws_qqp_dataset(path):
    # TODO(mm): there's probably a better way of doing this
    data_files = {"validation": path}
    dataset = load_dataset("csv", data_files=data_files, sep="\t")
    dataset = dataset["validation"]

    def _clean_data(sample):
        # the paws-qqp dataset was created as a stream of bytes. So every sentence starts with "b and ends with ".
        # we remove these
        sample["sentence1"] = sample["sentence1"][2:-1]
        sample["sentence2"] = sample["sentence2"][2:-1]
        return sample

    dataset = dataset.map(_clean_data, batched=False)
    dataset = dataset.rename_column("id", "idx")
    dataset = dataset.cast_column("label", ClassLabel(names=["yes", "no"]))

    label_list = dataset.features["label"].names
    num_labels = len(label_list)

    return dataset, label_list, num_labels

def get_dataset(data_set_used):
    datasets, labels, num_labels = load_glue_datasets(data_set_used)

    if data_set_used in ['mnli', 'rte', 'hans']:
        teacher_prompt = 'Think logically. Are the following sentences examples of entailment, yes or no?'
        student_prompt = 'Are the following sentences examples of entailment, yes or no?'
    elif data_set_used in ['qqp', 'paws-qqp']:
        teacher_prompt = 'Think logically. Are the following sentences duplicates or paraphrases of each other, yes or no?'
        student_prompt = 'Are the following sentences duplicates or paraphrases of each other, yes or no?'

    return datasets, labels, num_labels, teacher_prompt, student_prompt