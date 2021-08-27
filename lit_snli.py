# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:43:50 2021

@author: Shadow
"""

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, classification_report
from pytorch_lightning.callbacks import EarlyStopping
import os 

class LIT_SNLI(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 hidden_dropout_prob=.5,
                 attention_probs_dropout_prob=.2,
                 encoder_name = 'bert-base-uncased',
                 save_fp='best_model.pt'):
       
        super(LIT_SNLI, self).__init__()
        
        self.num_classes = num_classes
        
        self.build_model(hidden_dropout_prob, attention_probs_dropout_prob, encoder_name)
        
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.gt_probs, self.correctness = [], []
        
        self.save_fp = save_fp
    
    def build_model(self, hidden_dropout_prob, attention_probs_dropout_prob, encoder_name):
        config = AutoConfig.from_pretrained(encoder_name, num_labels=self.num_classes)
        #These are the only two dropouts that we can set
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder = AutoModelForSequenceClassification.from_pretrained(encoder_name, config=config)
        
    def save_model(self):
        torch.save(self.state_dict(), self.save_fp)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        # Run Forward Pass
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels = batch['labels'])

        # Compute Loss (Cross-Entropy)
        loss = outputs.loss
        
        # Getting logits 
        logits = outputs.logits
        logits = nn.functional.softmax(logits, dim=-1)
        
        #Get ground truth probabilities 
        ground_truth_probs = logits[0, batch['labels']]

        
        # Getting the predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Getting the correctness 
        correct = []
        for pred, label in zip(preds, batch['labels']):
            
            if pred == label:
                correct.append(True)
            else:
                correct.append(False)
        
        correct = torch.tensor(correct)
        
        # Compute Answer Accuracy
        preds = preds.detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        
        acc = accuracy_score(labels, preds)

        # Set up Data to be Logged
        return {"loss": loss, 'train_loss': loss, "ground_truth_probs": ground_truth_probs, "correct_mask": correct, 'train_acc':acc}

    def training_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.train_losses.append(avg_loss.detach().cpu())
        
        avg_acc = np.stack([x["train_acc"] for x in outputs]).mean()
        self.train_accs.append(avg_acc)
        
        #both of these have shape [# examples]
        gt_probs = torch.cat([x['ground_truth_probs'] for x in outputs])
        correctness = torch.cat([x['correct_mask'] for x in outputs])
        
        self.gt_probs.append(gt_probs.detach().cpu().numpy())
        self.correctness.append(correctness.detach().cpu().numpy())
        
        print('Train Loss: ', avg_loss)
        
    def validation_step(self, batch, batch_idx):

        # Run Forward Pass
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels = batch['labels'])
        
        # Compute Loss (Cross-Entropy)
        loss = outputs.loss
        
        # Compute Answer Accuracy
        logits = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        preds = preds.detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        acc = accuracy_score(labels, preds)
       
        return {"val_loss": loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        avg_loss_cpu = avg_loss.detach().cpu().numpy()
        if len(self.val_losses) == 0 or avg_loss_cpu<np.min(self.val_losses):
            self.save_model()
            
        self.val_losses.append(avg_loss_cpu)
        
        avg_acc =  np.stack([x["val_acc"] for x in outputs]).mean()
        self.val_accs.append(avg_acc)
        
        #self.log('val_loss', avg_loss, self.current_epoch)
        
        print('Val Loss: ', avg_loss)
        
        


def train_LitModel(model, train_data, val_data, max_epochs, batch_size, patience = 3, num_gpu=1):
    
    #
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=False)#, num_workers=8)#, num_workers=16)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle = False)
    
    #early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False, mode="min")
    
    trainer = pl.Trainer(gpus=num_gpu, max_epochs = max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    model.gt_probs, model.correctness = (np.array(model.gt_probs)).T, (np.array(model.correctness)).T
    model.train_losses, model.val_losses = np.array(model.train_losses), np.array(model.val_losses)

    return model


def model_testing(model, test_dataset):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    preds, total_labels = [], []
    
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        
        seq = (batch['input_ids']).to(device)
        mask = (batch['attention_mask']).to(device)
        labels = batch['labels']
        
        outputs = model(input_ids=seq, attention_mask=mask, labels=None)
        
        logits = outputs.logits
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        predictions = torch.argmax(logits, dim=-1)
        predictions = predictions.detach().cpu().numpy()
        
        preds.extend(predictions)
        total_labels.extend(labels)
        
    cr = classification_report(total_labels, preds, output_dict=True)
    return cr
        

class SNLI_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])