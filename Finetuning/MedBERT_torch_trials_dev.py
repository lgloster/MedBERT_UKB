from transformers import BertModel
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset, SubsetRandomSampler
import torch
import torch.nn as nn
from datasets import load_dataset
# from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter

# #custom metrics class
from MedBERT_metrics_trials import MedBERT_report
# from sklearn.model_selection import KFold
import json
import os
import sys
import pickle

##set recursion limit for json.dump(report_dict)
sys.setrecursionlimit(500000)

##SCRIPT ARGS
run_name = sys.argv[1]

###MAKING DIRECTORY FOR RUN DATA
output_dir = f".../{run_name}"
report_dir = output_dir+f"/{run_name}_run_reports"
tensorboard_dir = output_dir+f"/{run_name}_run_tensorboard"
weights_dir = output_dir+f"/{run_name}_run_weights"
roc_auc_dir = output_dir+f"/{run_name}_run_roc_auc_info"

os.mkdir(output_dir)
os.mkdir(report_dir)
os.mkdir(tensorboard_dir)
os.mkdir(weights_dir)
os.mkdir(roc_auc_dir)


#init device and writer for Tensorboard
device = torch.device("cuda:0")
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
writer = SummaryWriter(tensorboard_dir)

#RUN PARAMETERS
trials = 10
batch_size = 64
epochs = 10

###FILE DIRECTORIES
data_dir = ".../MITrials_json"

##LOADING & DEFINING MODEL

#loading pretrained model
medBERT = BertModel.from_pretrained(".../Finetuning")
# print(medBERT)

# freezing pretrained BERT layers
for param in medBERT.parameters():
  param.requires_grad = False

#finetuning model layers for binary classification
class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      # dropout layer
      self.dropout = nn.Dropout(0.05)

      # activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(192,64)

      # dense layer 2 (Hidden layer)
      self.fc2 = nn.Linear(64,16)

      # dense layer 3 (Hidden layer)
      self.fc3 = nn.Linear(16,1)

      #sigmoid activation function #note: not needed because of built-in BCEWithLogitsLoss
    #   self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask, segments):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, position_ids=segments, return_dict=False)
      #input layer 1
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      # hidden layer 1
      x = self.fc2(x)
      x = self.relu(x)
      # output layer 1
      x = self.fc3(x)
      # # apply sigmoid activation
      # x = self.sigmoid(x)
      
      return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(medBERT)
# model.load_state_dict(torch.load(".../MI_run_1_epoch0_saved_weights.pt"))
# push the model to GPU
model = model.to(device)

#optimizer---------------------------------------------
optimizer = AdamW(model.parameters(), lr = 5e-5)

report_dict = {}
###TRIALS: MAIN LOOP
for trial in range(1,trials+1):
    print("STARTING TRIAL:",trial, "#"*40)

    ###LOADING: DATA
    train_DATA = data_dir+f"/MI_Trial_{trial}_train.json"
    val_DATA = data_dir+f"/MI_Trial_{trial}_valid.json"

    # #small samples for dev testing
    # train_DATA = data_dir+f"/MI_Trial_{trial}_sample_train.json"
    # val_DATA = data_dir+f"/MI_Trial_{trial}_sample_valid.json"

    tokens_train = load_dataset('json', data_files=train_DATA)
    tokens_val = load_dataset('json', data_files=val_DATA)

    # for train set
    train_seq = torch.tensor(tokens_train['train']['input_ids'])
    train_mask = torch.tensor(tokens_train['train']['input_mask'])
    train_segment_ids = torch.tensor(tokens_train['train']['segment_ids'])
    train_y = torch.tensor(tokens_train['train']['next_sentence_labels'])

    # for validation set
    val_seq = torch.tensor(tokens_val['train']['input_ids'])
    val_mask = torch.tensor(tokens_val['train']['input_mask'])
    val_segment_ids = torch.tensor(tokens_val['train']['segment_ids'])
    val_y = torch.tensor(tokens_val['train']['next_sentence_labels'])

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_segment_ids, train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)
    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)
    
    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_segment_ids, val_y)
    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)
    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size, drop_last=True)

    #compute the class weights-----------------------------
    train_weight = (train_y==0.).sum()/(train_y==1.).sum()
    valid_weight = (val_y==0.).sum()/(val_y==1.).sum()

    print(f"TRIAL_{trial} TRAIN PENALIZATION WEIGHT:", train_weight)
    print(f"TRIAL_{trial} VALID PENALIZATION WEIGHT:",valid_weight)

    cross_entropy_train = nn.BCEWithLogitsLoss(pos_weight=train_weight)
    cross_entropy_val = nn.BCEWithLogitsLoss(pos_weight=valid_weight)

    #training----------------------------------------------
    def train():

        model.train()

        total_loss, total_accuracy = 0, 0

        # empty list to save model predictions
        total_preds=[]
        batch_labels = []
        # iterate over batches
        for step,batch in enumerate(train_dataloader):

            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            sent_id, mask, segments, labels = batch

            # clear previously calculated gradients 
            model.zero_grad()        

            # get model predictions for the current batch
            preds = model(sent_id, mask, segments) #Linear
            loss_preds = preds.type(torch.float32) 

            # compute the loss between actual and predicted values
            labels = labels.type(torch.float32)

            #loss function
            loss = cross_entropy_train(loss_preds, labels) #Linear
            # print(loss)
            a = loss.item()

            # add on to the total loss
            total_loss = total_loss + a

            # backward pass to calculate the gradients
            loss.backward()

            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            optimizer.step()

            # Sigmoid output of predictions
            sigmoid = nn.Sigmoid()
            preds = sigmoid(preds)
            preds = preds.type(torch.float32) #Sigmoid

            # model predictions are stored on GPU. So, push it to CPU
            preds=preds.detach().cpu().numpy()
            labels=labels.detach().cpu().numpy()

            # append the model predictions
            total_preds.append(preds)
            batch_labels.append(labels)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)

        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds  = np.concatenate(total_preds, axis=0)
        batch_labels = np.concatenate(batch_labels, axis=0)
  
        #returns the loss and predictions
        return avg_loss, total_preds, batch_labels

    #evaluation--------------------------------------------
    def evaluate():

        print("\nEvaluating...")

        # deactivate dropout layers
        model.eval()

        total_loss, total_accuracy = 0, 0

        # empty list to save the model predictions
        total_preds = []
        batch_labels = []

        # iterate over batches
        for step,batch in enumerate(val_dataloader):

            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # push the batch to gpu
            batch = [t.to(device) for t in batch]

            sent_id, mask, segments, labels = batch
            labels = labels.type(torch.float32)

            # deactivate autograd
            with torch.no_grad():
                
                # model predictions
                preds = model(sent_id, mask, segments)
                loss_preds = preds.type(torch.float32)

                # compute the validation loss between actual and predicted values
                loss = cross_entropy_val(loss_preds, labels)

                total_loss = total_loss + loss.item()

                #Sigmoid output
                sigmoid = nn.Sigmoid()
                preds = sigmoid(preds)
                preds = preds.type(torch.float32)

                preds=preds.detach().cpu().numpy()
                labels=labels.detach().cpu().numpy()

                # append the model predictions
                total_preds.append(preds)
                batch_labels.append(labels)

        # compute the validation loss of the epoch
        avg_loss = total_loss / len(val_dataloader) 

        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds  = np.concatenate(total_preds, axis=0)
        batch_labels = np.concatenate(batch_labels, axis=0)
      
        return avg_loss, total_preds, batch_labels


    #start_training loop----------------------------------------

    # set initial loss to infinite
    best_valid_loss = float('inf')
    best_train_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]
    report_dict = {}
    
    report_dict[f"Trial_{trial}"] = {}

    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        #train model
        train_loss, t_preds, t_labels = train()

        #evaluate model
        valid_loss, v_preds, v_labels = evaluate()

        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("Best valid epoch" + "-"*40, epoch, sep='\n')
            
        #print best epoch
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            print("Best train epoch" + "-"*40, epoch, sep='\n')

        #save model weights
        torch.save(model.state_dict(), f'{weights_dir}/{run_name}_Trial_{trial}_epoch_{epoch}_saved_weights.pt')
        #add train/val loss curve to Tensorboard
        writer.add_scalars(f'{run_name}_loss', {'train_loss': round(train_loss,3), 'val_loss': round(valid_loss,3)}, epoch)

        #instantiate metrics object with current epoch
        report = MedBERT_report(epoch)

        #sklearn_reports and class counts/epoch
        t_report, v_report, roc_file = report.classification_report(t_labels, t_preds, v_labels, v_preds, train_loss, valid_loss)
        counts = report.track_counts(t_labels, t_preds,v_labels, v_preds)

        #saving roc_file as pickled list (too long to fit into report_dict for JSON output)
        with open(roc_auc_dir+"/"+f"Trial_{trial}_Epoch_{epoch}_roc_curve", 'wb') as f:
            pickle.dump(roc_file, f)

        #updating report_dict with metrics/predictions from current epoch
        report_dict[f"Trial_{trial}"].update({f"Epoch {epoch}": counts})

        report_dict[f"Trial_{trial}"][f"Epoch {epoch}"]["Counts"]["valid"]["valid_report"] = v_report
        report_dict[f"Trial_{trial}"][f"Epoch {epoch}"]["Counts"]["valid"]["valid_loss"] = valid_loss

        report_dict[f"Trial_{trial}"][f"Epoch {epoch}"]["Counts"]["train"]["train_report"] = t_report
        report_dict[f"Trial_{trial}"][f"Epoch {epoch}"]["Counts"]["train"]["train_loss"] = train_loss

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    #clear cache and vars for new Trial dataset
    tokens_train.cleanup_cache_files()
    tokens_val.cleanup_cache_files()
    tokens_train = 0
    tokens_val = 0

    #converts np.int64 values from "classification_report" into int type for JSON saving
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError
    #save all epoch metrics in report_dict as JSON for current Trial 
    with open(f'{report_dir}/{run_name}_Trial_{trial}_report.json','w') as f:
        json.dump(report_dict, f, default=convert)
