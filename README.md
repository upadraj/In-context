# dl-project-llm

### Introduction

We run Efficient LLM Context Distillation model below:

### how to install: 

```
 conda env create -f relu_ranger.yaml
 conda activate relu_ranger
```

If in Visual Studio, you can CTRL + P => select python interpretert to select relu_ranger as your default environment.

Run jupyter notebook and open up the relevant file: opt-125m.ipynb or teacher_student.ipynb

Run all code to get model outputs.

### Interpretation:

The project implements context distillation by training a student model on a KL-divergence loss derived from a teacher model. Additionally, LoRA (Low-Rank Adaptation) is incorporated. 



### Models

- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-350m](https://huggingface.co/facebook/opt-350m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-13b](https://huggingface.co/facebook/opt-13b)
- [facebook/opt-30b](https://huggingface.co/facebook/opt-30b)


### Running the Model in Google Colab

Upload run_models.ipynb notebook and its dependencies (contained in context_utils.py, training_utils.py, and data_utils.py) on Google Colab.