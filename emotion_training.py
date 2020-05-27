"""# Libraries"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv

"""# Device"""

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model_path = '...'
model.load_state_dict(torch.load( model_path))
model = model.to(device)


class EmpathyDataset(Dataset):
  def __init__(self , empathy_path = '...'):
    super().__init__()

    self.dialogues = []
    csvfile = open(empathy_path)
    reader = csv.reader(csvfile)
    data = list(reader)
    tmp = []
    self.history = []
    self.reply = []

    for i in range(1,len(data)):
      if data[i][0] == '1':
        self.dialogues.append(tmp)
        tmp = []
      tmp.append(data[i][1])
    self.dialogues.pop(0)
    for dial in self.dialogues:
      self.history.append( "".join(dial[0:-1]))
      self.reply.append(dial[-1])

  def __len__(self):
    return len(self.dialogues)

  def __getitem__(self,item):
    return  "<|endoftext|> " + self.history[item] + self.reply[item]

dataset = EmpathyDataset()

empathy_loader = DataLoader(dataset, batch_size =1)

dataset.__getitem__(8)

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 3e-5
WARMUP_STEPS = 30
MAX_SEQ_LEN = 512
from transformers import AdamW, get_linear_schedule_with_warmup

model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
text_count = 0
sum_loss = 0.0
batch_count = 0
proc_seq_count = 0


models_folder = "..."
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for idx, convo in enumerate(empathy_loader):
  text = convo[0]
  #text_pair = convo[1]

print(text)
#print(text_pair)

for idx, convo in empathy_loader:
  print()

print(text)

for idx, convo in enumerate(empathy_loader):
  text = convo[0]
  text_pair = convo[1]

  tokenized_conversations = (tokenizer.encode_plus(text = text, text_pair = text_pair))
  attention_mask = torch.tensor(tokenized_conversations['attention_mask'],device=device)
  input_ids = torch.tensor(tokenized_conversations['input_ids'],device=device)
  token_type_ids = torch.tensor(tokenized_conversations['token_type_ids'],device=device)

  #next line can either be:
  outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids )
  break
print(outputs[1])



#Poetry Training Method:

for epoch in range(EPOCHS):

  print(f"EPOCH {epoch} started" + '=' * 30)
  for idx, convo in enumerate(empathy_loader):
    #print(idx)
    
    empathy_tens = torch.tensor(tokenizer.encode(convo[0], max_length=1020)).unsqueeze(0).to(device)

  #next line can either be:
    #outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids ) #convai
    outputs = model(empathy_tens, labels = empathy_tens )

  #next line can be either or:
  #loss = lm_loss #conva
    loss, logits = outputs[:2]
    #loss = outputs[0] #jokes

    loss.backward()

    sum_loss = sum_loss + loss.detach().data #jokes
   
    proc_seq_count = proc_seq_count +1
    if proc_seq_count == BATCH_SIZE:
      proc_seq_count = 0
      batch_count += 1
      optimizer.step()
      scheduler.step() #jokes ; may not be used with convai code 
      optimizer.zero_grad()
      
      model.zero_grad()

      if batch_count == 100:
        print("sum loss:" , sum_loss)
        batch_count = 0
        sum_loss = 0.0

#Saving the model
  torch.save(model.state_dict(), os.path.join(models_folder, "..."))

