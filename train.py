import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate
from datasets import load_dataset
from datasets import DownloadMode
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from google.colab import drive
from datasets import load_dataset, concatenate_datasets
import shutil
import numpy as np
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
import evaluate
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import random
import warnings
from transformers import DataCollatorForLanguageModeling
from sklearn.exceptions import UndefinedMetricWarning
from peft import LoraConfig, get_peft_model

DEFAULT_TEMPLATE = "Answer the question using only the given context(if can the question can not be answered do not print anything.). : \n\nContext: {} \n\nQuestion: {} \n\nAnswer: "
MAX_EPOCHS = 1
MAX_ITERATION = 1000
TRAIN_LOG_STEP = 10
RANDOM_SEED = 14
NUM_CLASSES = 2
VALIDATION_LOG_STEP = TRAIN_LOG_STEP * 5
RANGE_MIN = 12
RANGE_MAX = 14
EXPERIMENT_NAME = "/kaggle/working/gru_linear_probe_12_13_1000_maskcorrected"

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def split_dataset(dataset, validation_split=0.0001):
    # Split the train dataset into training and validation sets
    split_dataset = dataset.train_test_split(test_size=validation_split, seed = RANDOM_SEED)
    return split_dataset['train'], split_dataset['test']




def tokenize_fun(example):

    prompt = DEFAULT_TEMPLATE.format(
        example["context"],
        example["question"]
    )

    if len(example["answers"]["text"]) == 0:
        answer = ""
    else:
        answer = example["answers"]["text"][0]

    full_text = prompt + answer

    full = tokenizer(full_text, truncation=True)
    prompt_ids = tokenizer(prompt, truncation=True)["input_ids"]

    labels = full["input_ids"].copy()

    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels
    }


model_name = 'Qwen/Qwen3-0.6B'

# -----------------------
# Load dataset
# -----------------------

dataset = load_dataset("squad_v2",split="train",  download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
dataset_test = load_dataset("squad_v2",split="validation",  download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
train_dataset, validation_dataset = split_dataset(dataset)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name,dtype = torch.float16)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

train_dataset = train_dataset.map(tokenize_fun, remove_columns=train_dataset.column_names)
validation_dataset = validation_dataset.map(tokenize_fun, remove_columns=validation_dataset.column_names)
dataset_test = dataset_test.map(tokenize_fun, remove_columns=dataset_test.column_names)

max_length = 384
doc_stride = 128

lora_config = LoraConfig(
    r=8,                        # rank
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"    # important
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="/kaggle/working/lora_results",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    max_steps=2,
    weight_decay=0.01,
    logging_dir="/kaggle/working/lora_runs",
    logging_steps=100,
    fp16=True,
    report_to="tensorboard",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()