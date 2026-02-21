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
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

DEFAULT_TEMPLATE = "Answer the question using only the given context. : \n\nContext: {} \n\nQuestion: {} \n\nAnswer: "
YES_NO_TEMPLATE = "Is the Question answerable from Context? (Answer with yes and no) : \n\nContext: {} \n\nQuestion: {} \n\nAnswer: "
MAX_EPOCHS = 1
MAX_ITERATION = 1000
TRAIN_LOG_STEP = 10
RANDOM_SEED = 14
NUM_CLASSES = 2
VALIDATION_LOG_STEP = TRAIN_LOG_STEP * 5
RANGE_MIN = 12
RANGE_MAX = 14
EXPERIMENT_NAME = "/kaggle/working/gru_linear_probe_12_13_1000_maskcorrected"
LEARNING_RATE = 1e-3

# setting random seed for deterministic result
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def reset_train_dict():
  train_dict = {}
  train_dict['iter_cnt'] = 0
  train_dict['loss'] = 0
  train_dict['total_loss'] = 0
  train_dict['epoch'] = 0
  train_dict['batch_count'] = 0
  return train_dict

def split_dataset(dataset, validation_split=0.0005):
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
            add_generation_prompt=True
        ))

    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


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
    output = get_input_batch(batch, tokenizer, YES_NO_TEMPLATE)
    output['ids'] = [bat['id'] for bat in batch]
    output['labels'] = [get_ans(bat['answers']['text']) for bat in batch]
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


activations = {}
replace_activations = {}
replace_position = 0

def make_hook(name):
    def hook(module, input, output):
        if (name in activations) and (name in replace_activations) and (not replace_activations[name]):
            output[:,replace_position,:] = activations[name][:,replace_position,:].to(DEVICE)
            replace_activations[name] = True
        elif (name not in activations):
            activations[name] = output.detach()
    return hook

def zip_folder(folder_path, zip_name):
  shutil.make_archive(zip_name, 'zip', folder_path)



# Load model directly

if check_we_are_in_colab():
  model_name = 'Qwen/Qwen3-0.6B'
else:
  model_name = "./qwen0.6model/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,dtype = torch.float16)

model = model.eval()

dataset = load_dataset("squad_v2",split="train",  download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
dataset_test = load_dataset("squad_v2",split="validation",  download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
train_dataset, validation_dataset = split_dataset(dataset)
dataset_zeros = dataset.filter(lambda x: len(x['answers']['text']) == 0)
dataset_ones = dataset.filter(lambda x: len(x['answers']['text']) > 0).select(range(dataset_zeros.num_rows))

local_rank = int(os.environ["LOCAL_RANK"])

# Init process group
dist.init_process_group("nccl", rank=local_rank, world_size=2)

device = torch.device(f"cuda:{local_rank}")
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

sampler = DistributedSampler(dataset_test)
loader = DataLoader(dataset_test, batch_size=8, collate_fn=collate_fn, shuffle=False, sampler=sampler)
results = []
results_ids = []
labels_true = []
labels_pred = []
with torch.no_grad():
    for idx,batch_test in enumerate(loader):
        batch_test['input_ids'] = batch_test['input_ids'].to(device)
        results_ids += batch_test['ids']
        del batch_test['labels']
        del batch_test['ids']
        outputs = model.module.generate(**batch_test, max_new_tokens=30)
        results +=  tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if idx == 4:
            break