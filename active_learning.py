# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:55:00 2021

@author: Shadow
"""

import os
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from datasets import load_dataset
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer

from lit_snli import *

def main(cluster_eval=True, cluster_type=None, save_dir = 'active_learning_files'):
    
    total_train_ds, val_ds, test_ds = load_dataset('snli', split=['train[:25000]', 'validation','test'])
        
    total_train_labels = total_train_ds['label']
    val_labels = val_ds['label']
    test_labels = test_ds['label']
    
    total_train_text = [i + ' ' + j for i, j in zip(total_train_ds['premise'],total_train_ds['hypothesis'])]
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
    total_train_text, total_train_labels = filter_unlabeled(total_train_text, total_train_labels)
    val_text, val_labels = filter_unlabeled(val_text, val_labels)
    test_text, test_labels = filter_unlabeled(test_text, test_labels)
    
    
    #our oracle batch and init train size will be 1% of the data
    #percent_train = .01
    #init_train_size = np.floor(percent_train*len(total_train_text))
    init_train_size = 250
    oracle_batch = init_train_size
    
    print()
    print('Our inital train size is: ', init_train_size)
    
    unlabled_df = pd.DataFrame({'text':total_train_text,
                                'labels':total_train_labels})
    
    if cluster_eval == True:
        with open('cluster_region2d.pkl', 'rb') as f:
            cluster2d_regions = pickle.load(f)

        cluster_region = cluster2d_regions[cluster_type]
        mask = cluster_region.index.values
        cluster_data = unlabled_df.iloc[mask, :]
        
        unlabled_df = cluster_data
    
    print()
    print('The size of the unlabeled pool: ', len(unlabled_df))
    print()
    
    init_train = unlabled_df.sample(n=init_train_size, replace = False, random_state = 0)
    
    active_learning_iterations = 15
    
    encoder_name = 'bert-base-uncased'
        
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
    MAX_LEN = 64
    

    
    accs, macro_f1, macro_recall, macro_prec = [], [], [], []
    
    
    for step in range(active_learning_iterations):
        
        train_token = tokenizer(init_train['text'].tolist(), max_length = MAX_LEN, pad_to_max_length=True, truncation = True)
        val_token = tokenizer(val_text,max_length = MAX_LEN, pad_to_max_length=True, truncation = True)
        test_token = tokenizer(test_text, max_length = MAX_LEN, pad_to_max_length=True, truncation = True)    
        
        train_seq = torch.tensor(train_token['input_ids'])
        train_mask = torch.tensor(train_token['attention_mask'])
        train_y = torch.tensor(init_train['labels'].tolist())
        
        val_seq = torch.tensor(val_token['input_ids'])
        val_mask = torch.tensor(val_token['attention_mask'])
        val_y = (torch.tensor(val_labels))
        
        test_seq = torch.tensor(test_token['input_ids'])
        test_mask = torch.tensor(test_token['attention_mask'])
        test_y = torch.tensor(test_labels)
        
        
        train_enc = {'input_ids': train_seq, 'attention_mask': train_mask} 
        train_data = SNLI_Dataset(train_enc, train_y)
        
        val_enc = {'input_ids': val_seq, 'attention_mask': val_mask} 
        val_data = SNLI_Dataset(val_enc, val_y)
        
        test_enc = {'input_ids': test_seq, 'attention_mask': test_mask} 
        test_data = SNLI_Dataset(test_enc, test_y)
        
        #current_percent_train = str((step+1)*percent_train)
        save_file = 'bert_'+str(len(init_train))
        
        
        model = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.1, attention_probs_dropout_prob=.1, encoder_name=encoder_name, save_fp = save_dir+'/bert_train.pt')
        model = train_LitModel(model, train_data, val_data, max_epochs=10, batch_size=4, patience = 2, num_gpu=1)
        
        model = LIT_SNLI(num_classes = 3, hidden_dropout_prob=.1, attention_probs_dropout_prob=.1, encoder_name=encoder_name)
        model.load_state_dict(torch.load(save_dir+'/bert_train.pt'))
        
        cr = model_testing(model, test_data)
        
        print()
        print('Active Learning Iteration: ', step+1)
        print('Accuracy: ', cr['accuracy'])
        print()
        
        macro_f1.append(cr['macro avg']['f1-score'])
        macro_prec.append(cr['macro avg']['precision'])
        macro_recall.append(cr['macro avg']['recall'])
        accs.append(cr['accuracy'])
        
        '''
        with open(save_dir+'/'+save_file+'.pkl', 'wb') as f:
            pickle.dump(cr, f)
        '''
        #getting samples from oracle 
        oracle_samples = unlabled_df.sample(n=init_train_size, replace = False)
        
        init_train = pd.concat([init_train, oracle_samples], ignore_index=True)
    
    print('Sanity Check # Accs: ', len(accs))
    active_learning_stats = {'accs':accs,
                             'macro_f1':macro_f1,
                             'macro_prec':macro_prec,
                             'macro_recall':macro_recall}

    
    return active_learning_stats
            
if __name__=="__main__":
    
    
    save_dir = 'active_learning_files'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    runs = 3
    
    accs, f1, prec, recall = [], [], [], []
    for run in range(runs):
        
        print()
        print('##############')
        print('RUN #:',run+1)
        print('##############')
        
        stats = main(cluster_eval=True, cluster_type='hard', save_dir=save_dir)
        accs.append(stats['accs'])
        f1.append(stats['macro_f1'])
        prec.append(stats['macro_prec'])
        recall.append(stats['macro_recall'])
        
    accs, f1, prec, recall = np.array(accs), np.array(f1), np.array(prec), np.array(recall)
    avg_accs = np.mean(accs, axis=0)
    avg_f1 = np.mean(f1, axis=0)
    avg_prec = np.mean(prec, axis=0)
    avg_recall = np.mean(recall, axis=0)
    
    avg_active_learning_stat = {'accs':avg_accs,
                             'macro_f1':avg_f1,
                             'macro_prec':avg_prec,
                             'macro_recall':avg_recall}
    
    with open(save_dir+'/active_learning_hard_stats.pkl', 'wb') as f:
            pickle.dump(avg_active_learning_stat, f)