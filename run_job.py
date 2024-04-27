import os
import torch
import json
import random
import numpy as np
import json
import datasets as ds
import pandas as pd
import matplotlib.pyplot as plt

from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model
from random import randint, sample

from data_utils import (
    load_glue_datasets,
    load_hans_dataset,
    load_paws_qqp_dataset,
    get_dataset,
)
from context_utils import (
    create_few_shot_context, 
    select_demonstrations, 
    create_train_batch_token,
    create_validation_batch_token,
)
from training_utils import (
    set_seed,
    CastOutputToFloat,
    get_model,
    plot_losses,
    train,
    predict,
    run_job
)

def run_job(dataset_used, model_name, epochs, val_len, train_len, context_len, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, labels, num_labels, teacher_prompt, student_prompt = get_dataset(data_set_used)
    
    set_seed(seed)
    print('starting run: {}'.format(seed))
    print('loading model')
    tokenizer, student_model, teacher_model = get_model(model_name)
    print("finished loading model")

    print("loading data")
    train_data_tokens, train_data_strings, indices, context_indices = create_train_batch_token(
        data_set_used, 
        datasets, 
        teacher_description = teacher_prompt, 
        student_description = student_prompt, 
        tokenizer=tokenizer, 
        seed=seed, 
        device=device,
        num_shots = context_len,
        num_train_samps=train_len,
    )
    print("finished loading data")

    print("training model")
    train(teacher_model, student_model, train_data_tokens, epochs = epochs, device=device)
    print("finished training model")

    print("predicting on validation set")
    student_prompt_tokens, student_prompt_strings, val_indices, val_labels = create_validation_batch_token(
        data_set_used, datasets, prompt_descr=student_prompt ,tokenizer=tokenizer, device=device, limit=100
    )
    prediction = predict(student_model, student_prompt_tokens, tokenizer = tokenizer, device=device)  
    print("finished predicting on validation set")

    accuracy = accuracy_score(prediction,val_labels)
    print("finished run {}".format(seed))
    print("final result",accuracy)

    if not os.path.exists('output'):
        os.makedirs('output')
        
    meta_data_file_name = f'{dataset_used}_{model_name}_{seed}_{epochs}_{val_len}_{train_len}_{context_len}.json'
    metadata_loc = os.path.join('output',meta_data_file_name)
    metadata = {
        'accuracy': accuracy,
        'query_indices': indices,
        'context_indices': context_indices.tolist(),
        'validation_indices': val_indices.tolist(),
        'model_name': model_name,
        'dataset_used': dataset_used,
        'seed': seed,
        'epochs': epochs,
        'val_len': val_len,
        'train_len': train_len,
        'context_len': context_len
    }
    
    with open(metadata_loc, 'w') as f:
        json.dump(metadata, f)

    return None

seed = 2     
data_set_used = 'mnli'
model_name = "opt-125m"
epochs = 2
val_len = 128
train_len = 128
context_len = 16

run_job(data_set_used,model_name,epochs,val_len, train_len, context_len, seed)