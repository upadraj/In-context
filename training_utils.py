import os
import json
import random
import numpy as np
import torch
import datasets as ds
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model

from context_utils import (
    create_validation_batch_token,
    create_train_batch_token,
    create_hans_batch_token,
    create_paws_qqp_batch_token,
    create_qqp_validation_batch
)
from data_utils import (
    load_hans_dataset,
    load_paws_qqp_dataset,
    get_dataset,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds.logging.set_verbosity(40)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return {"trainable params":trainable_params, "all params": all_param}


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_model(model_name):
    model_name = "facebook/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # tokenizer
    model = OPTForCausalLM.from_pretrained(model_name)  # teacher model
    # student_model = OPTForCausalLM.from_pretrained(
    #     model_name, config=config
    # )  # student model

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    student_model = get_peft_model(model, config)

    student_model.config.hidden_dropout_prob = 0.1

    student_model.gradient_checkpointing_enable()
    student_model.enable_input_require_grads()
    student_model.lm_head = CastOutputToFloat(model.lm_head)
    
    return tokenizer, student_model, model

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker="o", linestyle="-", color="b")
    plt.title("Training Loss Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.show()


def train(teacher_model, student_model, data, epochs=10, batch_size=32, device="cpu"):
    student_model.to(device)
    teacher_model.to(device)

    student_model.train()
    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=1e-5, weight_decay=0.000001
    )
    epoch_losses = []

    if len(data) % batch_size != 0:
        num_batches = (len(data) // batch_size) + 1
    else:
        num_batches = len(data) // batch_size

    num_datapoints = len(data)
    total_steps = num_batches * epochs
    warmup_ratio = int(0.1 * total_steps)

    def lr_schedular(current_step: int):
        if current_step < warmup_ratio:
            return float(current_step) / float(max(1, warmup_ratio))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_schedular)

    for epoch in range(epochs):
        total_loss = 0
        samples_left = num_datapoints

        for i in range(num_batches):
            batch_loss = 0
            if (samples_left - batch_size) >= 0:
                samples_used = batch_size
                samples_left -= batch_size
            else:
                samples_used = samples_left
                samples_left = 0
            for j in range(samples_used):
                index = i * batch_size + j

                teacher_inputs = data[index]["context"].to(device)
                student_inputs = data[index]["query"].to(device)

                teacher_outputs = teacher_model.generate(
                    **teacher_inputs,
                    max_length=teacher_inputs["input_ids"].shape[-1] + 1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                teacher_probs = torch.nn.functional.softmax(
                    teacher_outputs.scores[0], dim=-1
                )

                student_logits = student_model(**student_inputs).logits
                student_probs = torch.nn.functional.softmax(
                    student_logits[:, -1, :], dim=-1
                )

                kl_divergence = torch.nn.functional.kl_div(
                    student_probs.log(), teacher_probs, reduction="batchmean"
                )

                optimizer.zero_grad()
                kl_divergence.backward()
                optimizer.step()

                batch_loss += kl_divergence.item()

            # Average loss for the batch
            batch_loss /= batch_size
            total_loss += batch_loss
            scheduler.step()

        # Average loss for the epoch
        epoch_loss = total_loss / num_batches
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}, Total Loss: {epoch_loss}")

    print(f"Total loss : {total_loss/epochs}")
    plot_losses(epoch_losses)


def predict(model, source, tokenizer, target=None, device="cpu"):
    predict = []
    for token in source:
        output = model.generate(
            **token, max_length=token["input_ids"].shape[-1] + 1
        ).to(device)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted_label = (
            decoded_output.split("Label:")[-1].strip().split(".")[0].strip()
        )
        predict.append(predicted_label)

    return predict


def get_validation_accuracy(
    student_model,
    student_prompt,
    tokenizer,
    datasets,
    dataset_used,
    device="cpu",
    val_len=1000,
    shuffle=False,
    indomain=True,
    seed=42
):

    if indomain:
        student_prompt_tokens, student_prompt_strings, val_indices, val_labels = (
            create_validation_batch_token(
                dataset_name=dataset_used,
                datasets=datasets,
                tokenizer=tokenizer,
                device=device,
                prompt_descr=student_prompt,
                limit=val_len,
                shuffle=shuffle,
            )
        )
    elif dataset_used == "qqp" and not indomain:
        student_prompt_tokens, student_prompt_strings, val_indices, val_labels = (
            create_paws_qqp_batch_token(
                datasets,
                tokenizer=tokenizer,
                device=device,
                prompt_descr=student_prompt,
                limit=val_len,
                shuffle=shuffle,
            )
        )
    elif dataset_used in ["mnli","rte"] and not indomain:
        student_prompt_tokens, student_prompt_strings, val_indices, val_labels = (
            create_hans_batch_token(
                datasets,
                tokenizer=tokenizer,
                device=device,
                prompt_descr=student_prompt,
                limit=val_len,
                shuffle=shuffle,
                seed=seed
            )
        )
    else:
        raise Exception("no dataset found for validation...")

    prediction = predict(
        student_model, student_prompt_tokens, tokenizer=tokenizer, device=device
    )

    accuracy = accuracy_score(prediction, val_labels)
    return val_indices, accuracy


def run_job(dataset_used, model_name, epochs, val_len, train_len, context_len, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, labels, num_labels, teacher_prompt, student_prompt = get_dataset(
        dataset_used
    )
    set_seed(seed)
    print("starting run: {}".format(seed))
    print("loading model")
    tokenizer, student_model, teacher_model = get_model(model_name)
    print("finished loading model")

    print("loading data")
    train_data_tokens, train_data_strings, indices, context_indices, train_labels = (
        create_train_batch_token(
            dataset_used,
            datasets,
            teacher_description=teacher_prompt,
            student_description=student_prompt,
            tokenizer=tokenizer,
            seed=seed,
            device=device,
            num_shots=context_len,
            num_train_samps=train_len,
        )
    )
    print("finished loading data")

    print("training model")
    train(
        teacher_model,
        student_model,
        train_data_tokens.copy(),
        epochs=epochs,
        device=device,
    )
    print("finished training model")

    print("predicting teacher on validation set")
    t_predict = predict(
        teacher_model,
        [tdt["context"] for tdt in train_data_tokens],
        tokenizer=tokenizer,
        device=device,
    )
    t_accuracy = accuracy_score(t_predict, train_labels)
    print("finished run in-domain teacher val{}".format(seed))
    print("final result", t_accuracy)

    print("predicting in-domain on validation set")
    indom_val_index, indom_accuracy = get_validation_accuracy(
        student_model=student_model,
        student_prompt=student_prompt,
        tokenizer=tokenizer,
        datasets=datasets,
        dataset_used=dataset_used,
        device=device,
        val_len=val_len,
    )
    print("finished run in-domain val{}".format(seed))
    print("final result", indom_accuracy)
    
    print("predicting student model out-domain on validation set of qqp ")
    qqp_validation_token, val_labels = create_qqp_validation_batch(tokenizer, device, limit=50)
    prediction = predict(
        student_model, qqp_validation_token, tokenizer=tokenizer, device=device
    )

    out_domin = accuracy_score(prediction, val_labels)
    print("finished run out-domain val{}".format(seed))
    print("final result", out_domin )
    
    if dataset_used == "qqp":
        print("predicting on paws_qqp")
        validation_set = "paws_qqp"

        dataset, label_list, num_labels = load_paws_qqp_dataset("dev_and_test.tsv")

        ood_val_index, ood_accuracy = get_validation_accuracy(
            student_model=student_model,
            student_prompt=student_prompt,
            tokenizer=tokenizer,
            datasets=dataset,
            dataset_used=dataset_used,
            device=device,
            val_len=val_len,
            indomain=False,
        )
        print("finished run ood {}".format(seed))
        print("final result", ood_accuracy)
    else:
        print("predicting on hans")
        dataset, label_list, num_labels = load_hans_dataset()
        validation_set = "hans"

        ood_val_index, ood_accuracy = get_validation_accuracy(
            student_model=student_model,
            student_prompt=student_prompt,
            tokenizer=tokenizer,
            datasets=dataset,
            dataset_used=dataset_used,
            device=device,
            val_len=val_len,
            shuffle=True,
            indomain=False,
            seed=seed,
        )
        print("finished run ood {}".format(seed))
        print("final result", ood_accuracy)
    meta_domain_name = "out"

    if not os.path.exists("output"):
        os.makedirs("output")

    meta_data_file_name = f"{dataset_used}_{model_name}_{seed}_{meta_domain_name}_{epochs}_{val_len}_{train_len}_{context_len}.json"
    metadata_loc = os.path.join("output", meta_data_file_name)
    metadata = {
        "accuracy": {
            "teacher": t_accuracy,
            "indomain": indom_accuracy,
            "outdomain": ood_accuracy,
        },
        "query_indices": indices,
        "context_indices": context_indices.tolist(),
        "validation_indices": {
            "indomain": indom_val_index.tolist(),
            "outdomain": ood_val_index.tolist(),
        },
        "model_name": model_name,
        "dataset_used": dataset_used,
        "validation_used": validation_set,
        "indomain": meta_domain_name,
        "seed": seed,
        "epochs": epochs,
        "val_len": val_len,
        "train_len": train_len,
        "context_len": context_len,
        "params": print_trainable_parameters(student_model),

    }

    with open(metadata_loc, "w") as f:
        json.dump(metadata, f)

    return None
