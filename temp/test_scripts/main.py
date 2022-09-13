#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('nvidia-smi')


# In[2]:


import os
import pandas as pd
import tqdm
import math


# In[3]:


# from google.colab import drive

# drive.mount('/content/drive', force_remount=True)


# In[4]:


# from google.colab import auth
# auth.authenticate_user()

# import gspread
# from oauth2client.client import GoogleCredentials

# gc = gspread.authorize(GoogleCredentials.get_application_default())


# In[5]:


# worksheet = gc.open('Tweets_Spring_Summer_2021_coded').sheet1

# sheet_data = worksheet.get_all_values()
# df = pd.DataFrame.from_records(sheet_data[1:], columns=sheet_data[0])
# # print('Num examples:', len(df))
# # print('Null Values\n', df.isna().sum())
# # df.dropna(inplace=True)
# # print('Num examples:', len(df))

# # in the tweets find the hashtag
# df['hashTags'] = df['Tweet'].str.findall("#(\w+)")
# # in the tweets find the mentions
# df['mentions'] = df['Tweet'].str.findall("@(\w+)")

# # Remove hashtag and mentions
# df['Tweet'] = df['Tweet'].str.replace(r'#(\w+)', '', regex=True)
# df['Tweet'] = df['Tweet'].str.replace(r'@(\w+)', '', regex=True)

# df


# In[6]:


#df1 =pd.read_csv('20161006.csv',lineterminator='\n',skipinitialspace=True, usecols= ['text'])
#df2 =pd.read_csv('20161007.csv',lineterminator='\n',skipinitialspace=True, usecols= ['text'])
df =pd.read_csv('2020_01_01.csv',lineterminator='\n',skipinitialspace=True, usecols= ['text'])


# frame = [df1, df2]
# df = pd.concat(frame)
df.rename(columns={'text':'Tweet'}, inplace=True)
df.shape


# In[7]:


df = df.dropna()


# In[8]:


df.shape


# In[9]:


#Set the path to the data folder, datafile and output folder and files
root_folder = '/users/kent/jmaharja/drugAbuse'
# data_folder = os.path.abspath(os.path.join(root_folder, 'datasets/text_gen_product_names'))
model_folder = os.path.abspath(os.path.join(root_folder, 'output/Drug-Abuse/RoBERTaMLM/'))
output_folder = os.path.abspath(os.path.join(root_folder, 'output/Drug-Abuse'))
tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'output/Drug-Abuse/TokRoBERTa/'))

# test_filename='Tweets_Spring_Summer_2021_coded'
# datafile= 'product_names_desc_cl_train.csv'
outputfile = 'submission.csv'

# datafile_path = os.path.abspath(os.path.join(data_folder,datafile))
# testfile_path = os.path.abspath(os.path.join(data_folder,test_filename))
outputfile_path = os.path.abspath(os.path.join(output_folder,outputfile))


# Build a Tokenizer

# In[10]:


# Drop the files from the output dir
# txt_files_dir = "./text_split_2020"
# !rm -rf {txt_files_dir}
# !mkdir {txt_files_dir}


# In[11]:


# Store values in a dataframe column (Series object) to files, one file per record
def column_to_files(column, prefix, txt_files_dir):
    # The prefix is a unique ID to avoid to overwrite a text file
    i=prefix
    #For every value in the df, with just one column
    for row in column.to_list():
      # Create the filename using the prefix ID
      file_name = os.path.join(txt_files_dir, str(i)+'.txt')
      try:
        # Create the file and write the column text to it
        f = open(file_name, 'wb')
        f.write(row.encode('utf-8'))
        f.close()
      except Exception as e:  #catch exceptions(for eg. empty rows)
        print(row, e) 
      i+=1
    # Return the last ID
    return i


# In[12]:


data = df["Tweet"]
# Removing the end of line character \n
data = data.replace("\n"," ")
# Set the ID to 0
prefix=0
# Create a file for every description value
# prefix = column_to_files(data, prefix, txt_files_dir)
# # Print the last ID
# print(prefix)


# Train the tokenizer
# 

# In[13]:


from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import torch
from torch.utils.data.dataset import Dataset


# In[14]:


#get_ipython().run_cell_magic('time', '', 'paths = [str(x) for x in Path(".").glob("text_split/*.txt")]\n\n# Initialize a tokenizer\ntokenizer = ByteLevelBPETokenizer(lowercase=True)\n\n# Customize training\ntokenizer.train(files=paths, vocab_size=8192, min_frequency=2,\n                show_progress=True,\n                special_tokens=[\n                                "<s>",\n                                "<pad>",\n                                "</s>",\n                                "<unk>",\n                                "<mask>",\n])')
#time 
paths = [str(x) for x in Path(".").glob("text_split/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=8192, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])

# In[ ]:


#Save the Tokenizer to disk
tokenizer.save_model(tokenizer_folder)


# In[15]:


# Create the tokenizer using vocab.json and mrege.txt files
tokenizer = ByteLevelBPETokenizer(
    os.path.abspath(os.path.join(tokenizer_folder,'vocab.json')),
    os.path.abspath(os.path.join(tokenizer_folder,'merges.txt'))
)


# In[16]:


# Prepare the tokenizer
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)


# In[17]:


tokenizer.encode("cook some blue.")


# In[18]:


tokenizer.encode("cook some blue.").tokens


# Train a language model from scratch
# 
# 

# In[19]:


TRAIN_BATCH_SIZE = 16    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 8    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 15        # number of epochs to train (default: 10)
LEARNING_RATE = 1e-4    # learning rate (default: 0.001)
WEIGHT_DECAY = 0.01
SEED = 42               # random seed (default: 42)
MAX_LEN = 128
SUMMARY_LEN = 7


# In[20]:


# Check that PyTorch sees it
import torch
torch.cuda.is_available()


# In[21]:


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=8192,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


# In[22]:


from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print('Num parameters: ',model.num_parameters())


# In[23]:


from transformers import RobertaTokenizerFast
# Create the tokenizer from a trained one
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=MAX_LEN)


# In[24]:


from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
train_df, test_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=RANDOM_SEED)


# Building the training Dataset

# In[25]:


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


# In[26]:


# Create the train and evaluation dataset
train_dataset = CustomDataset(train_df['Tweet'], tokenizer)
eval_dataset = CustomDataset(test_df['Tweet'], tokenizer)


# In[27]:


from transformers import DataCollatorForLanguageModeling

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# In[28]:


from torch import nn
from transformers import Trainer, TrainingArguments


# In[29]:


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


# In[ ]:


# Train the model
trainer.train()


# In[ ]:


eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# In[ ]:


trainer.save_model(model_folder)


# In[ ]:


tokenizer_folder


# In[ ]:


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=model_folder,
    tokenizer=tokenizer_folder
)


# In[ ]:


fill_mask("Alcohol and drugs is good for the <mask>")


# In[ ]:





