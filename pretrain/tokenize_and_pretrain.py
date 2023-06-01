#!/usr/bin/env python
# coding: utf-8

# In[2]:


#nvidia-smi

# In[3]:


import os
import pandas as pd
import tqdm
import math


# In[7]:


#Set the path to the data folder, datafile and output folder and files
root_folder = '/data/jmharja/robertaForTweetAnalysis'
model_folder = os.path.abspath(os.path.join(root_folder, 'output/RoBERTaMLM/'))
output_folder = os.path.abspath(os.path.join(root_folder, 'output/'))
tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'output/TokRoBERTa/'))

datafile= '2020_01_01.csv'
# testfile= '20161007.csv'
# outputfile = 'submission.csv'

datafile_path = os.path.abspath(os.path.join(root_folder,'input/'+ datafile))
# testfile_path = os.path.abspath(os.path.join(root_folder,'input/'+ testfile))
# outputfile_path = os.path.abspath(os.path.join(output_folder, outputfile))


# In[9]:

train_df =pd.read_csv(datafile_path,lineterminator='\n',skipinitialspace=True, usecols= ['text'])
train_df.rename(columns={'text':'Tweet'}, inplace=True)
train_df = train_df.dropna()
train_df.shape


# # Build a Tokenizer

# In[10]:


# Drop the files from the output dir
# txt_files_dir = "./text_split_2020"
# get_ipython().system('rm -rf {txt_files_dir}')
# get_ipython().system('mkdir {txt_files_dir}')


# In[11]:


# Store values in a dataframe column (Series object) to files, one file per record
# The prefix is a unique ID to avoid to overwrite a text file
# def column_to_files(column, prefix, txt_files_dir):
#     i=prefix
#     for row in column.to_list():
#       file_name = os.path.join(txt_files_dir, str(i)+'.txt')
#       try:
#         f = open(file_name, 'wb')
#         f.write(row.encode('utf-8'))
#         f.close()
#       except Exception as e: 
#         print(row, e) 
#       i+=1
#     return i


# In[12]:


# data = train_df["Tweet"]
# data = data.replace("\n"," ")
# prefix = 0
# #Create a file for every description value
# prefix = column_to_files(data, prefix, txt_files_dir)
# print(prefix)


# In[14]:


from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
from torch.utils.data.dataset import Dataset


# In[15]:


# paths = [str(x) for x in Path(".").glob("text_split_2020/*.txt")]


# In[16]:


#Save the Tokenizer to disk
# tokenizer.save_model(tokenizer_folder)


# In[17]:


# Create the tokenizer using vocab.json and mrege.txt files
# tokenizer = ByteLevelBPETokenizer(
#     os.path.abspath(os.path.join(tokenizer_folder,'vocab.json')),
#     os.path.abspath(os.path.join(tokenizer_folder,'merges.txt'))
# )


# # In[18]:


# # Prepare the tokenizer
# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=512)



# # Train a language model from scratch

# In[24]:


TRAIN_BATCH_SIZE = 16    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 8    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 32        # number of epochs to train (default: 10)
LEARNING_RATE = 1e-4    # learning rate (default: 0.001)
WEIGHT_DECAY = 0.01
SEED = 42               # random seed (default: 42)
MAX_LEN = 128
SUMMARY_LEN = 7


# In[25]:


# Check that PyTorch sees it
import torch
torch.cuda.is_available()


# In[26]:


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=8192,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


# In[27]:


from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)
print('Num parameters: ', model.num_parameters())
# if args.n_gpu > 1:
# model = torch.nn.DataParallel(model)


# In[28]:


from transformers import RobertaTokenizerFast
# Create the tokenizer from a trained one
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=MAX_LEN)


# In[29]:


df = train_df


# In[30]:


from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
train_df, test_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=RANDOM_SEED)


# # Building the training Dataset

# In[31]:


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []
        
        for example in df.values:
            x=tokenizer.encode_plus(example, max_length = MAX_LEN, truncation=True, padding=True)
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


# In[32]:


# Create the train and evaluation dataset
train_dataset = CustomDataset(train_df['Tweet'], tokenizer)
eval_dataset = CustomDataset(val_df['Tweet'], tokenizer)



# In[45]:


from transformers import DataCollatorForLanguageModeling
# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


# In[46]:


from torch import nn
from transformers import Trainer, TrainingArguments


# In[47]:


#from transformers import Trainer, TrainingArguments
print(model_folder)
# Define the training arguments
training_args = TrainingArguments(
    output_dir=model_folder,
    overwrite_output_dir=True,
    evaluation_strategy = 'epoch',
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    save_steps=8192,
    #eval_steps=4096,
    save_total_limit=1,
)
# Create the trainer for our model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #prediction_loss_only=True,
)


trainer.train()


eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.save_model(model_folder)


