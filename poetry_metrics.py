

"""Sentiment Analysis"""

import csv
gen_path = 'path to csv file with Generated Samples'
csvfile = open(gen_path)
reader = csv.reader(csvfile)
data = list(reader)

for i in range (1, len(data)):
  print(data[i][3])

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

for i in range (1, len(data)):
  print("SET1 ", sid.polarity_scores(data[i][6]))

"""For perplexity , XLNet is utilized"""



import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel
import numpy as np
from scipy.special import softmax

PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> """

for i in range (1, len(data)):
  text = data[i][4]
  
  tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
  model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

  tokenize_input = tokenizer.tokenize(PADDING_TEXT + text)
  tokenize_text = tokenizer.tokenize(text)

  sum_lp = 0.0
  for max_word_id in range((len(tokenize_input)-len(tokenize_text)), (len(tokenize_input))):

      sent = tokenize_input[:]

      input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sent)])

      perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
      perm_mask[:, :, max_word_id:] = 1.0 

      target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
      target_mapping[0, 0, max_word_id] = 1.0

      with torch.no_grad():
          outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
          next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

      word_id = tokenizer.convert_tokens_to_ids([tokenize_input[max_word_id]])[0]
      predicted_prob = softmax(np.array(next_token_logits[0][-1]))
      lp = np.log(predicted_prob[word_id])

      sum_lp += lp

  print("SET 2 poem:" ,i, "sentence logprob =", sum_lp)