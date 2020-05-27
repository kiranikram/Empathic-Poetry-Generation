import logging
logging.getLogger().setLevel(logging.CRITICAL)

import torch
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

###Load the pretrained model according to emotion set required 

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model_path = 'load path to trained model'
model.load_state_dict(torch.load(model_path))
model = model.to(device)

input_ids = torch.tensor(tokenizer.encode(context)).unsqueeze(0).long().to(device)
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=250, 
    top_k=50,
    temperature=0.7,
    top_p=0.95, 
    num_return_sequences=2
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

# Function to first select topN tokens from the probability list and then based on the selected N word distribution
# get random token ID
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def generate_some_text(input_str, text_len = 250):

    cur_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():

        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=10) #Randomly(from the given probability distribution) choose the next word from the top n words
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        print(output_text)