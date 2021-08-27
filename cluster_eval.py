# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:08:22 2021

@author: Shadow
"""

import os 
import pickle
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

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

directory ='bert_25k_small_drop'

train_df = pd.DataFrame({'text':train_text,
                         'labels':train_labels})

'''
with open(directory+'/'+'cluster_region2d.pkl', 'rb') as f:
    cluster2d_regions = pickle.load(f)
'''

with open('cluster_region2d.pkl', 'rb') as f:
    cluster2d_regions = pickle.load(f)

easy_region = cluster2d_regions['easy']
easy_mask = easy_region.index.values
easy_data = train_df.iloc[easy_mask, :]

print()
print('# of easy examples: ', len(easy_data))

ambig_region = cluster2d_regions['ambig']
ambig_mask = ambig_region.index.values
ambig_data = train_df.iloc[ambig_mask, :]

print()
print('# of ambig examples: ', len(ambig_data))

hard_region = cluster2d_regions['hard']
hard_mask = hard_region.index.values
hard_data = train_df.iloc[hard_mask,:]

print()
print('# of hard examples: ', len(hard_data))

#
encoder_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(encoder_name)

MAX_LEN = 64

train_token_easy = tokenizer(easy_data['text'].tolist(), max_length = MAX_LEN, pad_to_max_length=True, truncation = True)
train_token_ambig = tokenizer(ambig_data['text'].tolist(), max_length = MAX_LEN, pad_to_max_length=True, truncation = True)
train_token_hard = tokenizer(hard_data['text'].tolist(), max_length = MAX_LEN, pad_to_max_length=True, truncation = True)

val_token = tokenizer(val_text, max_length = MAX_LEN, pad_to_max_length=True, truncation = True)
test_token = tokenizer(test_text, max_length = MAX_LEN, pad_to_max_length=True, truncation = True)


import torch

train_seq_easy = torch.tensor(train_token_easy['input_ids'])
train_mask_easy = torch.tensor(train_token_easy['attention_mask'])
train_y_easy = torch.tensor(easy_data['labels'].tolist())

train_seq_ambig = torch.tensor(train_token_ambig['input_ids'])
train_mask_ambig = torch.tensor(train_token_ambig['attention_mask'])
train_y_ambig = torch.tensor(ambig_data['labels'].tolist())

train_seq_hard = torch.tensor(train_token_hard['input_ids'])
train_mask_hard = torch.tensor(train_token_hard['attention_mask'])
train_y_hard = torch.tensor(hard_data['labels'].tolist())



val_seq = torch.tensor(val_token['input_ids'])
val_mask = torch.tensor(val_token['attention_mask'])
val_y = (torch.tensor(val_labels))

test_seq = torch.tensor(test_token['input_ids'])
test_mask = torch.tensor(test_token['attention_mask'])
test_y = torch.tensor(test_labels)

from lit_snli import *

train_enc_easy = {'input_ids': train_seq_easy, 'attention_mask': train_mask_easy} 
train_data_easy = SNLI_Dataset(train_enc_easy, train_y_easy)

train_enc_ambig = {'input_ids': train_seq_ambig, 'attention_mask': train_mask_ambig} 
train_data_ambig = SNLI_Dataset(train_enc_ambig, train_y_ambig)

train_enc_hard = {'input_ids': train_seq_hard, 'attention_mask': train_mask_hard} 
train_data_hard = SNLI_Dataset(train_enc_hard, train_y_hard)

val_enc = {'input_ids': val_seq, 'attention_mask': val_mask} 
val_data = SNLI_Dataset(val_enc, val_y)

test_enc = {'input_ids':test_seq, 'attention_mask':test_mask}
test_data = SNLI_Dataset(test_enc, test_y)

if not os.path.exists('cluster2d_models'):
    os.makedirs('cluster2d_models')
    
model_easy = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name, save_fp = 'cluster2d_models/bert_easy2d_train.pt')
model_easy = train_LitModel(model_easy, train_data_easy, val_data, max_epochs=10, batch_size=8, patience = 3, num_gpu=1)
print()

model_ambig = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name, save_fp = 'cluster2d_models/bert_ambig2d_train.pt')
model_ambig = train_LitModel(model_ambig, train_data_ambig, val_data, max_epochs=10, batch_size=8, patience = 3, num_gpu=1)
print()

model_hard = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name, save_fp = 'cluster2d_models/bert_hard2d_train.pt')
model_hard = train_LitModel(model_hard, train_data_hard, val_data, max_epochs=10, batch_size=8, patience = 3, num_gpu=1)
print()

#saving the train stats
if not os.path.exists('cluster2d_train_stats'):
    os.makedirs('cluster2d_train_stats')
    
easy_files = {'train_losses': model_easy.train_losses, 
              'val_losses':model_easy.val_losses, 
              'train_accs': model_easy.train_accs,
              'val_accs': model_easy.val_accs}

ambig_files = {'train_losses': model_ambig.train_losses, 
              'val_losses':model_ambig.val_losses, 
              'train_accs': model_ambig.train_accs,
              'val_accs': model_ambig.val_accs}

hard_files = {'train_losses': model_hard.train_losses, 
              'val_losses':model_hard.val_losses, 
              'train_accs': model_hard.train_accs,
              'val_accs': model_hard.val_accs}

#saving training/testing statistics
with open('cluster2d_train_stats/easy_files.pkl', 'wb') as f:
    pickle.dump(easy_files, f)
    
with open('cluster2d_train_stats/ambig_files.pkl', 'wb') as f:
    pickle.dump(ambig_files, f)

with open('cluster2d_train_stats/hard_files.pkl', 'wb') as f:
    pickle.dump(hard_files, f)

model_easy = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name)
model_easy.load_state_dict(torch.load('cluster2d_models/bert_easy2d_train.pt'))
easy_cr = model_testing(model_easy, test_data)

model_ambig = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name)
model_ambig.load_state_dict(torch.load('cluster2d_models/bert_ambig2d_train.pt'))
ambig_cr = model_testing(model_ambig, test_data)


model_hard = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.3, attention_probs_dropout_prob=.2, encoder_name=encoder_name)
model_hard.load_state_dict(torch.load('cluster2d_models/bert_hard2d_train.pt'))
hard_cr = model_testing(model_hard, test_data)

#saving the train stats
if not os.path.exists('cluster2d_test_stats'):
    os.makedirs('cluster2d_test_stats')


#saving training/testing statistics
with open('cluster2d_test_stats/easy_files.pkl', 'wb') as f:
    pickle.dump({'cr':easy_cr}, f)
    
with open('cluster2d_test_stats/ambig_files.pkl', 'wb') as f:
    pickle.dump({'cr':ambig_cr}, f)

with open('cluster2d_test_stats/hard_files.pkl', 'wb') as f:
    pickle.dump({'cr':hard_cr}, f)


    