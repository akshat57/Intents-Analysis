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
    
    phone_to_idx = {'unk': 1}#Padding indx = 0, unkown_idx = 1, indexing starts from 2
    for i, phone in enumerate(vocab):
        phone_to_idx[phone] = i + 2
        
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


def get_domains():
    all_intents = ['increase', 'decrease', 'activate', 'deactivate', 'bring', 'change language']
    return all_intents

def get_intents():
    all_intents = [
        'activate|lamp',
        'activate|lights|bedroom',
        'activate|lights|kitchen',
        'activate|lights|none',
        'activate|lights|washroom',
        'activate|music',
        'bring|juice',
        'bring|newspaper',
        'bring|shoes',
        'bring|socks',
        'change language|Chinese',
        'change language|English',
        'change language|German',
        'change language|Korean',
        'change language|none',
        'deactivate|lamp',
        'deactivate|lights|bedroom',
        'deactivate|lights|kitchen',
        'deactivate|lights|none',
        'deactivate|lights|washroom',
        'deactivate|music',
        'decrease|heat|bedroom',
        'decrease|heat|kitchen',
        'decrease|heat|none',
        'decrease|heat|washroom',
        'decrease|volume',
        'increase|heat|bedroom',
        'increase|heat|kitchen',
        'increase|heat|none',
        'increase|heat|washroom',
        'increase|volume'
        ]

    return all_intents

def get_intent_labels(class_type):
    if class_type == 'domain':
        all_intents = get_domains()
    else:
        all_intents = get_intents()
        
    intent_labels = {}
    labels_to_intents = {}
    for i, intent in enumerate(all_intents):
        intent_labels[intent] = i
        labels_to_intents[i] = intent
        
    return intent_labels, labels_to_intents


# In[9]:


class_type = 'intents'
split = 'train'

intent_labels, labels_to_intents = get_intent_labels(class_type)

#Loading data
train_file = '../FSC/fsc_' + class_type + '_' + split + '.pkl'
test_file = '../FSC/fsc_' + class_type + '_test.pkl'
#create vocabulary and phone_to_idx
phone_to_idx = create_vocabulary(train_file)
print(len(phone_to_idx))


# In[10]:


train_dataset = MyDataset(train_file, intent_labels, phone_to_idx)
train_loader_args = dict(shuffle=True, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=True, batch_size=64)
train_loader = DataLoader(train_dataset, **train_loader_args, collate_fn=collate_indic)

test_dataset = MyDataset(test_file, intent_labels, phone_to_idx)
test_loader_args = dict(shuffle=False, batch_size=128, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=False, batch_size=32)
valid_loader = DataLoader(test_dataset, **test_loader_args, collate_fn=collate_indic)


# In[11]:


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size=50, embed_size=128, hidden_size=256, label_size=31):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(embed_size, embed_size, kernel_size=5, padding=2)
#        self.cnn3 = nn.Conv1d(embed_size, embed_size, kernel_size=7, padding=3)

        self.batchnorm = nn.BatchNorm1d(embed_size*2)

        self.lstm = nn.LSTM(embed_size*2, hidden_size, num_layers=1)
        self.linear = nn.Linear(hidden_size, label_size)

    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """

        # B,T,H
        input = self.embed(x)

        # (B,T,H) -> (B,H,T)
        input = input.transpose(1,2)

        #pass through CNN to capture Ngram features
        cnn_output = torch.cat([self.cnn(input), self.cnn2(input)], dim=1)
        embeddings = cnn_output##saved for attention
        input = F.relu(self.batchnorm(cnn_output))
        input = input.transpose(1,2)

        #Pass through LSTM
        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        _, (hn, cn) = self.lstm(pack_tensor)
        
        #ATTENTION BEGINS
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        batch_size = hn[-1].shape[0]
        emb_size = embeddings.shape[1]
        seq_len = embeddings.shape[2]
        
        attention = torch.zeros([batch_size, seq_len]).to(device)
        for t in range(seq_len):
            ####use this for cosine similarity attention
            #attention[:,t] = cos(embeddings[:,:,t], hn[-1])
            ####Use this for dot attentioin
            attention[:,t] = torch.diagonal(torch.matmul(embeddings[:,:,t], hn[-1].T))

                                                          
        #NORMALIZATION OF ATTENTION WEIGHTS
        #Normalizing by sum. Usually goes along with cosine attention
        #normalize = torch.sum(attention, dim=1).reshape(-1,1)
        #attention = attention/normalize
        
        #Dot product normalization
        attention /= (256**0.5)

        output = torch.zeros([batch_size, emb_size]).to(device)
        for t in range(seq_len):
            weights = attention[:,t].reshape(-1,1)
            output += weights*embeddings[:,:,t]

        logits = self.linear(hn[0])

        return logits


# In[12]:


model = RNNClassifier()
opt = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
#opt = SGD(model.parameters(), lr=0.05)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)


# In[ ]:


print(class_type, split)
max_acc = 0

for j in range(1000):
    #print("epoch ", i)
    loss_accum = 0.0
    batch_cnt = 0

    acc_cnt = 0
    err_cnt = 0

    model.train()
    start_time = time.time()
    for batch, (x, lengths, y) in enumerate(train_loader):

        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        opt.zero_grad()

        logits = model(x, lengths)

        loss = criterion(logits, y)
        loss_score = loss.cpu().item()

        loss_accum += loss_score
        batch_cnt += 1
        loss.backward()
        opt.step()

        out_val, out_indices = torch.max(logits, dim=1)
        tar_indices = y

        for i in range(len(out_indices)):
            if out_indices[i] == tar_indices[i]:
                acc_cnt += 1
            else:
                err_cnt += 1
                    

    print("train acc: ", acc_cnt/(err_cnt+acc_cnt), " train loss: ", loss_accum / batch_cnt, '--time:', time.time() - start_time)

    model.eval()
    acc_cnt = 0
    err_cnt = 0

    #start_time = time.time()
    for x, lengths, y in valid_loader:
        
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        
        logits = model(x, lengths)

        out_val, out_indices = torch.max(logits, dim=1)
        tar_indices = y

        for i in range(len(out_indices)):
            if out_indices[i] == tar_indices[i]:
                acc_cnt += 1
            else:
                err_cnt += 1

    current_acc = acc_cnt/(err_cnt+acc_cnt)
    if current_acc > max_acc:
        max_acc = current_acc
                
    print(j, "validation: ", current_acc, '--max', max_acc, '--time:', time.time() - start_time)


# In[ ]:




