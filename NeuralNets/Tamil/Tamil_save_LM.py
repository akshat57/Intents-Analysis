#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
#import psutil
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import random
import sys
import pickle
from torch.optim import SGD


# In[2]:


sys.path.insert(1, '/home/akshatgu/Intents-Analysis/Analysis')
#sys.path.insert(1, '/Users/manjugupta/Desktop/CMU_Courses/Intents/getting_intents/Analysis')


# In[3]:


from get_vocab import load_data, get_vocab
from get_frequency import get_frequency


# In[4]:


#Check if cuda is available
cuda = torch.cuda.is_available()
print('CUDA is', cuda)
CUDA_LAUNCH_BLOCKING=1

num_workers = 8 if cuda else 0

print(num_workers)


# In[5]:


##Needed Functions
def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output


def create_vocabulary(train_file):
    '''This function creates an indexed vocabulary dictionary from the training file'''
    
    vocab, _ = get_vocab(1, train_file)
    
    phone_to_idx = {'eos':1, 'unk': 2}#Padding indx = 0, eos = 1, unkown_idx = 2, indexing starts from 3
    for i, phone in enumerate(vocab):
        phone_to_idx[phone] = i + 3
        
    return phone_to_idx


# In[6]:


class MyDataset(Dataset):
    def __init__(self, data_file, intent_labels, phone_to_idx):
        data = load_data(data_file)
        self.all_data = []
        
        for intent in data:
            for utterance in data[intent]:
                if len(utterance) != 0:
                    utterance_to_idx = []

                    for phone in utterance:
                        if phone not in phone_to_idx:
                            phone = 'unk'

                        utterance_to_idx.append(phone_to_idx[phone])

                    self.all_data.append([utterance_to_idx, intent_labels[intent]])
            
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,index):
        input_vector = self.all_data[index][0]
        label = self.all_data[index][1]

        return input_vector, label


# In[7]:


def collate_indic(tuple_lst):

    x_lst = [x[0] for x in tuple_lst]
    y_lst = [x[1] for x in tuple_lst]

    # collate x
    B = len(tuple_lst)#Number of training samples
    T = max(len(x) for x in x_lst)#Max length of a sentence

    # x values
    x = torch.zeros([B, T], dtype=torch.int64)
    lengths = torch.zeros(B, dtype=torch.int64)

    for i, x_np in enumerate(x_lst):
        lengths[i] = len(x_np)
        x[i,:len(x_np)] = torch.tensor(x_np)

    # collate y
    y = torch.tensor(y_lst)

    ids = torch.argsort(lengths, descending=True)

    return x[ids], lengths[ids], y[ids]


# In[8]:


class MyLMDataset(Dataset):
    def __init__(self, data_file, intent_labels, phone_to_idx):
        data = load_data(data_file)
        self.all_data = []
        
        for intent in data:
            for utterance in data[intent]:
                if len(utterance) != 0:
                    utterance_to_idx = []

                    for phone in utterance:
                        if phone not in phone_to_idx:
                            phone = 'unk'

                        utterance_to_idx.append(phone_to_idx[phone])

                    self.all_data.append([utterance_to_idx, intent_labels[intent]])
            
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,index):
        input_vector = self.all_data[index][0]
        output_vector = self.all_data[index][0][1:] + [1]

        return input_vector, output_vector


# In[9]:


def collate_LM(tuple_lst):

    x_lst = [x[0] for x in tuple_lst]
    y_lst = [x[1] for x in tuple_lst]
    

    # collate x
    B = len(tuple_lst)#Number of training samples
    T = max(len(x) for x in x_lst)#Max length of a sentence

    # x values
    x = torch.zeros([B, T], dtype=torch.int64)
    y = torch.zeros([B, T], dtype=torch.int64)
    lengths = torch.zeros(B, dtype=torch.int64)

    for i, x_np in enumerate(x_lst):
        lengths[i] = len(x_np)
        x[i,:len(x_np)] = torch.tensor(x_np)
        y[i,:len(x_np)] = torch.tensor(y_lst[i])


    ids = torch.argsort(lengths, descending=True)

    return x[ids], lengths[ids], y[ids]


# In[10]:


def get_intents():
    all_intents = ['1', '2', '3', '4', '5', '6']
    return all_intents

def get_intent_labels(class_type):
    all_intents = get_intents()
        
    intent_labels = {}
    labels_to_intents = {}
    for i, intent in enumerate(all_intents):
        intent_labels[intent] = i
        labels_to_intents[i] = intent
        
    return intent_labels, labels_to_intents


# In[34]:


class_type = 'intents'

intent_labels, labels_to_intents = get_intent_labels(class_type)

#Loading data
split = '1'
train_file = '../../Tamil_Dataset/datasplit1/tamil_train_split_' + split + '.pkl'
test_file = '../../Tamil_Dataset/datasplit1/tamil_test_split_' + split + '.pkl'
#create vocabulary and phone_to_idx
phone_to_idx = create_vocabulary(train_file)
vocab_size = len(phone_to_idx) + 1
print(vocab_size)


# In[35]:


train_dataset = MyLMDataset(train_file, intent_labels, phone_to_idx)
train_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=True, batch_size=64)
train_loader_LM = DataLoader(train_dataset, **train_loader_args, collate_fn=collate_LM)


# In[36]:


train_dataset = MyDataset(train_file, intent_labels, phone_to_idx)
train_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=True, batch_size=64)
train_loader = DataLoader(train_dataset, **train_loader_args, collate_fn=collate_indic)

test_dataset = MyDataset(test_file, intent_labels, phone_to_idx)
test_loader_args = dict(shuffle=False, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=False, batch_size=64)
valid_loader = DataLoader(test_dataset, **test_loader_args, collate_fn=collate_indic)


# In[66]:


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, label_size=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(embed_size, embed_size, kernel_size=5, padding=2)
        #self.cnn3 = nn.Conv1d(embed_size, embed_size, kernel_size=7, padding=3)

        self.batchnorm = nn.BatchNorm1d(embed_size*2)

        self.lstm_LM = nn.LSTM(embed_size, hidden_size, num_layers=1)
        self.lstm = nn.LSTM(embed_size*2, hidden_size, num_layers=1)

        self.linear_LM = nn.Linear(hidden_size, vocab_size)
        self.linear = nn.Linear(hidden_size, label_size)

    def forward(self, x, lengths, lm = True):
        """
        padded_x: (B,T) padded LongTensor
        """
        # B,T,H
        
        input = self.embed(x)
        
        if lm:
            pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
            output, (hn, cn) = self.lstm_LM(pack_tensor)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            logits = self.linear_LM(output)

        else:

            # (B,T,H) -> (B,H,T)
            input = input.transpose(1,2)

            #cnn_output = torch.cat([self.cnn(input), self.cnn2(input), self.cnn3(input)], dim=1)
            cnn_output = torch.cat([self.cnn(input), self.cnn2(input)], dim=1)

            # (B,H,T)
            input = F.relu(self.batchnorm(cnn_output))

            input = input.transpose(1,2)

            pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
            output, (hn, cn) = self.lstm(pack_tensor)

            logits = self.linear(hn[0])

        return logits


# In[70]:


model = RNNClassifier(vocab_size)
opt = optim.Adam(model.parameters(), lr = 0.001)
#scheduler = MultiStepLR(optimizer, milestones=[50,80], gamma=0.1)
criterion = nn.CrossEntropyLoss()
#opt = SGD(model.parameters(), lr=0.01)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)


# In[71]:


#Train language model
print(class_type, split)
max_acc = 0

for j in range(200):
    
    #print("epoch ", i)
    loss_accum = 0.0
    batch_cnt = 0

    model.train()
    start_time = time.time()
    for batch, (x, lengths, y) in enumerate(train_loader_LM):

        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        opt.zero_grad()

        logits = model(x, lengths)
        
        loss = criterion(logits.permute(0,2,1), y)
        loss_score = loss.cpu().item()

        loss_accum += loss_score
        batch_cnt += 1
        loss.backward()
        opt.step()
                    
    print(j, " train loss: ", loss_accum / batch_cnt, '--time:', time.time() - start_time)


    if j == 9:
        torch.save(model, 'LM_10.pth')
    elif j == 49:
        torch.save(model, 'LM_50.pth')
    elif j == 99:
        torch.save(model, 'LM_100.pth')
    elif j == 199:
        torch.save(model, 'LM_200.pth')
