"""# Libraries"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
from io import open
import csv
import math
from torch.utils.tensorboard import SummaryWriter

"""# Device"""

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

class PoetryDataset(Dataset):
  def __init__(self , poetry_path = '/cpath'):
    super().__init__()

    PF_Poetry = []
    txtfile = open(poetry_path)
    poetry_data = txtfile.readlines()
    poetry_tmp = []
    self.POETRY = []
    Poem_String = ""

    for i in range(len(poetry_data)):
      if poetry_data[i] == '<|endoftext|>\n':
        PF_Poetry.append(poetry_tmp)
        poetry_tmp = []
      poetry_tmp.append(poetry_data[i])
        
    PF_Poetry.pop(0)
    
    for i in range(len(PF_Poetry)):
      Poem_String = "".join(PF_Poetry[i])
      if len(Poem_String.split())>300 : 
          Poem_String = ' '.join(Poem_String.split()[:300])
          
      self.POETRY.append(Poem_String)
      Poem_String = ""

    txtfile.close()

  def __len__(self):
    return len(self.POETRY)

  def __getitem__(self,item):
    #return  self.PF_Poetry[item]

    return self.POETRY[item]

dataset = PoetryDataset()

poetry_loader = DataLoader(dataset, batch_size =1)

dataset.__getitem__(17)

next(iter(poetry_loader))

"""# Training"""

batch = 16
no_of_epochs = 50
l_rate = 3e-5
steps_WU = 30
max_seq_len = 1020
from transformers import AdamW, get_linear_schedule_with_warmup

model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=l_rate)
learning = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_WU, num_training_steps = -1)
poetry_iterations = 0
losses = 0.0
batch_count = 0
proc_seq_count = 0


models_folder = "..."
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

tb = SummaryWriter(log_dir = 'Finetuning V1 Poetry')

#Training:

for epoch in range(no_of_epochs):

  print(f"EPOCH {epoch} started" + '=' * 30)
  for idx, poem in enumerate(poetry_loader):
    #print(idx)
    
    poetry_tens = torch.tensor(tokenizer.encode(poem[0], max_length=1020)).unsqueeze(0).to(device)

  #next line can either be:
    #outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids ) #convai
    outputs = model(poetry_tens, labels = poetry_tens )

  #next line can be either or:
  #loss = lm_loss 
    loss, logits = outputs[:2]
    #loss = outputs[0] 

    loss.backward()

    losses = losses + loss.detach().data 
   
    proc_seq_count = proc_seq_count +1
    if proc_seq_count == batch:
      proc_seq_count = 0
      batch_count += 1
      optimizer.step()
      learning.step() 
      optimizer.zero_grad()
      
      model.zero_grad()

      if batch_count == 100:
        print("sum loss:" , sum_loss)
        batch_count = 0
        sum_loss = 0.0


#Saving the model
  torch.save(model.state_dict(), os.path.join(models_folder, "gpt2_medium_V2poetry_{epoch}.pt"))