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

def reset_train_dict():
  train_dict = {}
  train_dict['iter_cnt'] = 0
  train_dict['loss'] = 0
  train_dict['total_loss'] = 0
  train_dict['epoch'] = 0
  train_dict['batch_count'] = 0
  return train_dict

def split_dataset(dataset, validation_split=0.0001):
    # Split the train dataset into training and validation sets
    split_dataset = dataset.train_test_split(test_size=validation_split, seed = RANDOM_SEED)
    return split_dataset['train'], split_dataset['test']

def count_number_of_tokens(sentence):
    return len(sentence.split())

def format_input_text(context, question, template=DEFAULT_TEMPLATE):
    prompt = template.format(context, question)
    return {"role": "user", "content": prompt}

def get_input(context, question, tokenizer, template=DEFAULT_TEMPLATE):
    inputs = tokenizer.apply_chat_template(
	[format_input_text(context, question)],
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
    enable_thinking=False,
	return_tensors="pt",)
    return inputs

def check_max_iteration(train_dict, step_number):
  if step_number < train_dict['iter_cnt']:
    return True
  return False

def check_step(train_dict, step_number):
  if ((train_dict['iter_cnt'] % step_number) == 0):
    return True
  return False

def get_input_batch(batch_input, tokenizer, template=DEFAULT_TEMPLATE):
    texts = []
    for bat in batch_input:
        texts.append(tokenizer.apply_chat_template(
            [format_input_text(bat['context'], bat['question'], template)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ))

    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

def get_output_batch(batch_input, tokenizer, template=DEFAULT_TEMPLATE):
    answers = []
    for bat in batch_input:
        if len(bat['answers']['text']) > 0:
            answers.append(tokenizer.apply_chat_template(
                [format_input_text(bat['answers']['text'][0], template)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ))
        else:
           answers.append(tokenizer.apply_chat_template(
                '.',
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ))

    return tokenizer(answers, padding=True, truncation=True, return_tensors="pt")

def get_labels(batch_input):
    labels = []
    for answer in batch_input:
        if len(answer['answers']['text']) == 0:
            labels.append(0)
        else:
            labels.append(1)
    return torch.tensor(labels)

def check_we_are_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
    


def get_summary_writer():
  if check_we_are_in_colab():
    writer = SummaryWriter('/tensor_board_logs/')
  else:
    writer = SummaryWriter(f'/share/tensor_board_logs/{EXPERIMENT_NAME}')
  return writer

def get_true_and_pred(test_dataset, classifier, criterion):
    labels_true = []
    labels_pred = []
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=collate_fn, shuffle=False, generator = get_new_generator())
    for batch_test in test_loader:
      input_ids_batch, attention_mask_batch, labels = batch_test['input_ids'].to(DEVICE), batch_test['attention_mask'].to(DEVICE), batch_test['labels'].to(DEVICE)
      labels_true += labels.tolist()
      with torch.no_grad():
        logits = classifier(input_ids_batch, attention_mask_batch)
        if criterion != None:
          test_loss = criterion(logits, labels)
        labels_pred += logits.argmax(dim=1).tolist()
    return np.array(labels_true), np.array(labels_pred)

def print_train_log(train_dict, layer_num):
    print(f"Epoch {train_dict['epoch']+1}/{MAX_EPOCHS} - Iter {train_dict['iter_cnt']} - Running Loss layer {layer_num}: {train_dict['total_loss']/(train_dict['batch_count']+1):.4f}")

def log_train(writer, train_dict, layer_num):
    writer.add_scalars("Loss Train", {f'layer{layer_num}' : train_dict['loss']}, train_dict['iter_cnt'])
    writer.add_scalars("Loss(total) Train", {f'layer{layer_num}' : train_dict['total_loss']}, train_dict['iter_cnt'])

def answer_question(context, question, tokenizer, model, print_tokes=False):
  inputs = get_input(context, question, tokenizer)
  if print_tokes:
    print_tokens_with_pos(inputs)
  inputs.to(model.device)
  outputs = model.generate(**inputs.to(model.device),
                         max_new_tokens=30)
  return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

def get_ans(answers_list):
    if (len(answers_list) > 0):
        return answers_list[0]
    else:
        return ' '

def collate_fn(batch):
    output = get_input_batch(batch, tokenizer, DEFAULT_TEMPLATE)
    output['ids'] = [bat['id'] for bat in batch]
    output['questions'] = [bat['question'] for bat in batch]
    output['labels'] = get_output_batch(batch, tokenizer)
    return output

def print_tokens_with_pos(inputs):
  print('token positions:')
  tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
  for token_idx, token in enumerate(tokens):
      if token_idx % 6 == 0: 
          print(f'{token_idx} -- {token}      ')
      else: 
          print(f'{token_idx} -- {token}    ', end = "")
  print('')



def causal_lm_collator(batch, tokenizer, max_length=512):
    """
    batch: list of raw examples with keys:
        - context
        - question
        - answers: {"text": [...], "answer_start": [...]}
    tokenizer: PreTrainedTokenizerBase
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for ex in batch:
        # Tokenize context + question
        tok = tokenizer(
            format_input_text(ex["context"], ex["question"], DEFAULT_TEMPLATE),
            truncation=True,
            padding_side='left',
            max_length=max_length,
            return_offsets_mapping=True
        )
        ids = tok["input_ids"]
        offsets = tok["offset_mapping"]

        # Take the first answer
        answer_text = ex["answers"]["text"][0]
        answer_start_char = ex["answers"]["answer_start"][0]
        answer_end_char = answer_start_char + len(answer_text)

        # Find answer token positions
        start_token = None
        end_token = None
        for i, (s, e) in enumerate(offsets):
            if s <= answer_start_char < e:
                start_token = i
            if s < answer_end_char <= e:
                end_token = i
        if start_token is None or end_token is None:
            # fallback: put answer at end (rare)
            start_token = len(ids) - 1
            end_token = len(ids) - 1

        # Build labels: -100 for context + question, token IDs for answer
        labels = [-100] * len(ids)
        for i in range(start_token, end_token + 1):
            labels[i] = ids[i]

        # Pad to max_length
        pad_len = max_length - len(ids)
        ids = ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * len(tok["input_ids"]) + [0] * pad_len

        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    batch_out = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list)
    }
    return batch_out


model_name = 'Qwen/Qwen3-0.6B'

# -----------------------
# Load dataset
# -----------------------

dataset = load_dataset("squad_v2",split="train",  download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
dataset_test = load_dataset("squad_v2",split="validation",  download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
train_dataset, validation_dataset = split_dataset(dataset)
dataset_zeros = dataset.filter(lambda x: len(x['answers']['text']) == 0)
dataset_ones = dataset.filter(lambda x: len(x['answers']['text']) > 0).select(range(dataset_zeros.num_rows))


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name,dtype = torch.float16)

max_length = 384
doc_stride = 128

def preprocess(example):

    tokenized = tokenizer(
        format_input_text(example["context"], example["question"]),
        truncation="only_second",
        padding_side='left',
        max_length=max_length,
        stride=doc_stride,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offsets = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offsets):

        answer = example["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        start_token = context_start
        end_token = context_end

        for idx in range(context_start, context_end+1):
            if offset[idx][0] <= start_char <= offset[idx][1]:
                start_token = idx
            if offset[idx][0] <= end_char <= offset[idx][1]:
                end_token = idx

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)



metric = evaluate.load("squad")

def postprocess(predictions, examples):

    start_logits, end_logits = predictions

    results = []

    for i in range(len(start_logits)):

        start = np.argmax(start_logits[i])
        end = np.argmax(end_logits[i])

        input_ids = examples[i]["input_ids"]

        answer = tokenizer.decode(
            input_ids[start:end+1],
            padding_side='left',
            skip_special_tokens=True
        )

        results.append(answer)

    return results

def compute_metrics(eval_pred):

    start_logits, end_logits = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = []
    references = []

    for i in range(len(start_logits)):

        start = np.argmax(start_logits[i])
        end = np.argmax(end_logits[i])

        input_ids = dataset["validation"][i]["input_ids"]

        pred_text = tokenizer.decode(
            input_ids[start:end+1],
            padding_side='left',
            skip_special_tokens=True
        )

        predictions.append({
            "id": dataset["validation"][i]["id"],
            "prediction_text": pred_text
        })

        references.append({
            "id": dataset["validation"][i]["id"],
            "answers": dataset["validation"][i]["answers"]
        })

    return metric.compute(
        predictions=predictions,
        references=references
    )

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
    task_type="QUESTION_ANS"    # important
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

# -----------------------
# Trainer
# -----------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=causal_lm_collator,
    compute_metrics=compute_metrics,
)

trainer.train()