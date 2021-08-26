# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:45:34 2021

@author: Shadow
"""


from datasets import load_dataset

train_ds, val_ds, test_ds = load_dataset('snli', split=['train[:25000]', 'validation','test'])

train_labels = train_ds['label']
val_labels = val_ds['label']
test_labels = test_ds['label']

train_text = [i + ' ' + j for i, j in zip(train_ds['premise'],train_ds['hypothesis'])]
val_text = [i + ' ' + j for i, j in zip(val_ds['premise'], val_ds['hypothesis'])]
test_text = [i + ' ' + j for i, j in zip(test_ds['premise'],test_ds['hypothesis'])]

def filter_unlabeled(texts, labels):
    
    filtered_text, filtered_labels = [], []
 
    for text, label in zip(texts, labels):
        
        if label != -1:
            filtered_text.append(text)
            filtered_labels.append(label)
    
    return filtered_text, filtered_labels

#Filtering out the the unlabeled instances
train_text, train_labels = filter_unlabeled(train_text, train_labels)
val_text, val_labels = filter_unlabeled(val_text, val_labels)
test_text, test_labels = filter_unlabeled(test_text, test_labels)

from transformers import BertForSequenceClassification
from transformers import AutoTokenizer

encoder_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(encoder_name)

MAX_LEN = 64

train_token = tokenizer(train_text, max_length = MAX_LEN, pad_to_max_length=True, truncation = True)
val_token = tokenizer(val_text,max_length = MAX_LEN, pad_to_max_length=True, truncation = True)
test_token = tokenizer(test_text, max_length = MAX_LEN, pad_to_max_length=True, truncation = True)



#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 3)

import torch

train_seq = torch.tensor(train_token['input_ids'])
train_mask = torch.tensor(train_token['attention_mask'])
train_y = torch.tensor(train_labels)

val_seq = torch.tensor(val_token['input_ids'])
val_mask = torch.tensor(val_token['attention_mask'])
val_y = (torch.tensor(val_labels))

test_seq = torch.tensor(test_token['input_ids'])
test_mask = torch.tensor(test_token['attention_mask'])
test_y = torch.tensor(test_labels)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#needa define a batch size
#batch_size = 32

from lit_snli import *


train_enc = {'input_ids': train_seq, 'attention_mask': train_mask} 
train_data = SNLI_Dataset(train_enc, train_y)

val_enc = {'input_ids': val_seq, 'attention_mask': val_mask} 
val_data = SNLI_Dataset(val_enc, val_y)

test_enc = {'input_ids': test_seq, 'attention_mask': test_mask} 
test_data = SNLI_Dataset(test_enc, test_y)


model = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name, save_fp = 'bert_25k_small_batch.pt')

model = train_LitModel(model, train_data, val_data, max_epochs=6, batch_size=4, patience = 5, num_gpu=2)

gt_probs = model.gt_probs
correctness = model.correctness

import os 

if not os.path.exists('bert_25k_small_batch'):
    os.makedirs('bert_25k_small_batch')
    
#Saving train statistics

train_statistics = {'gt_probs': model.gt_probs,
                    'correctness':model.correctness,
                    'train_losses':model.train_losses,
                    'val_losses':model.val_losses,
                    'train_accs':model.train_accs,
                    'val_accs':model.val_accs}

import pickle

with open('bert_25k_small_batch/bert_25k_train_stats.pkl', 'wb') as f:
    pickle.dump(train_statistics, f)

#reloading the model for testing
model = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name)

model.load_state_dict(torch.load('bert_25k_small_batch.pt'))

cr = model_testing(model, test_data)

with open('bert_25k_small_batch/bert_25k_test_stats.pkl', 'wb') as f:
    pickle.dump(cr, f)

