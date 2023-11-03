from transformers import get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizerFast, RobertaModel, RobertaPreTrainedModel
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from collections import defaultdict
from textwrap import wrap
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import re, os, traceback, sys


# %matplotlib inline
# %config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

hyperparams = {
    'BATCH_SIZE': 16,
    'EPOCHS': 32,
    'RANDOM_SEED': 42,
    'MAX_LEN' : 128,
    'lr' : 2e-5,
    'cuda' : 'cuda:0'
}

np.random.seed(hyperparams['RANDOM_SEED'])
torch.manual_seed(hyperparams['RANDOM_SEED'])
device = torch.device(hyperparams['cuda'] if torch.cuda.is_available() else "cpu")
current_time = datetime.now().strftime("%Y%m%d-%I_%M%p")
# outFilepath = 'out/drug_classifier/' + device.type + current_time +'/'
# try:
#     os.makedirs(outFilepath)
#     print(outFilepath)
# except FileExistsError:
#     pass

# Create the tokenizer from a trained one
tokenizer_folder = '/data/jmharja/projects/robertaForTweetAnalysis/output/oct2022/TokRoBERTa'
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=hyperparams['MAX_LEN'])

class Tweet_DataSet(Dataset):
   def __init__(self, data, tokenizer, max_len):
    self.data = data
    self.data['Tweet'] = self.data['Tweet'].map(lambda x: self.cleaner(x))
    self.tokenizer = tokenizer
    self.max_len = max_len
    
  
   def __len__(self):
    return len(self.data)

   def cleaner(self, tweet):
#         print(tweet)
        tweet = re.sub("@[A-Za-z0-9]+","", tweet) #Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
        tweet = " ".join(tweet.split())
        #     tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
        #     tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        #     tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
        #          if w.lower() in words or not w.isalpha())
        return tweet
    
        
  
   def __getitem__(self, index:int):
    data_row = self.data.iloc[index]
    tweet = data_row.Tweet
    labels = data_row['label']
    encoding = tokenizer.encode_plus(tweet,
                                     None,
                                     max_length = hyperparams['MAX_LEN'],
                                     truncation=True,
                                     pad_to_max_length=True,
                                     add_special_tokens=True,
                                     padding='max_length',
                                     return_token_type_ids=True)

    return {
      'tweet_text': tweet,
      'input_ids': torch.tensor(encoding.input_ids, dtype=torch.long),
      'attention_mask':  torch.tensor(encoding.attention_mask, dtype=torch.long),
      'token_type_ids': torch.tensor(encoding.token_type_ids, dtype=torch.long),
      'targets': torch.tensor(labels, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = Tweet_DataSet(df,tokenizer=tokenizer,max_len=max_len)
  return DataLoader(ds, batch_size=batch_size,num_workers=4)

class TweetModel(RobertaPreTrainedModel):
    def __init__(self, conf, n_classes):
        super(TweetModel, self).__init__(conf)
        self.roberta = RobertaModel.from_pretrained('/data/jmharja/projects/robertaForTweetAnalysis/output/oct2022/RoBERTaMLM/', config=conf)
        self.drop_out = torch.nn.Dropout(0.5)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.classifier = torch.nn.Linear(768, n_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.drop_out(pooler)
        output = self.classifier(pooler)
        return output
    
config = RobertaConfig(
    vocab_size=8192,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
    hidden_size=768,
    pad_token_id=1
)

model = TweetModel(config, 2)
model = model.to(device)
model.load_state_dict(torch.load('/data2/julina/checkpoint/drugAbuseFineTuneCkpt/best_ftc_model_state2023_02_02-05_16PM.bin'), strict=False)

def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:
      texts = d["tweet_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      token_type_ids = d["token_type_ids"].to(device)
      targets = d["targets"].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
      _, preds = torch.max(outputs, dim=1)
      probs = torch.nn.functional.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values


# def show_confusion_matrix(y_pred, y_test,  filename):
#     cm = confusion_matrix(y_pred, y_test)
#     df_cm = pd.DataFrame(cm, index=['Y', 'N'], columns=['Y', 'N'])
#     hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
#     hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
#     hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
#     plt.ylabel('Actual ')
#     plt.xlabel('Predicted ')
#     plt.savefig(filename + '_cm.png')
    
def predict_for_folder(input_folder, output_folder):
    outFilepath = 'out/drug_classifier/' + device.type + current_time +'/'
    try:
        os.makedirs(output_folder)
        print('created' + output_folder)
    except FileExistsError:
        pass
    
    input_files = os.listdir(input_folder)
    for file in input_files:
        try:
            filename = input_folder + file
            output_filename = output_folder + file
            if output_filename in os.listdir(output_folder):
                print("Done:: ", output_filename)
                continue
             
            df_pred =pd.read_csv(filename ,lineterminator='\n',skipinitialspace=True,)
            print(df_pred.shape)
            df_pred.drop(df_pred.columns[[0, 1]], axis=1, inplace=True)
            df_pred = df_pred.rename(columns={"text": 'Tweet'})
            df_pred['label']= 1
            # df_pred = df_pred[:100] # Test for 100
            pred_data_loader = create_data_loader(df_pred, tokenizer, hyperparams['MAX_LEN'],  hyperparams['BATCH_SIZE'])
            y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, pred_data_loader)
            
            y_pred_probs_pd = [y.numpy() for y in y_pred_probs]
            someListOfLists = list(zip(y_review_texts, y_pred.numpy()))
            npa = np.asarray(someListOfLists)
            dff = pd.DataFrame(someListOfLists, columns = ['Tweet', 'DrugAbuse' ])
            ones = dff[dff['DrugAbuse']==1]['Tweet']
            print (ones.size)
            if ones.size > 1:
                ones.to_csv(output_filename + 'pred.csv')
            
            # show_confusion_matrix(y_pred, y_test, output_filename)
            cm = confusion_matrix(y_pred, y_test)
            df_cm = pd.DataFrame(cm, index=['Y', 'N'], columns=['Y', 'N'])
            hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
            hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
            hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
            plt.ylabel('Actual ')
            plt.xlabel('Predicted ')
            plt.savefig(output_filename + '_cm.png')
            plt.figure().clear()
            print(f"Done for {file}!" )
               
        except:
            traceback.print_exc()
            pass
        

if __name__ == "__main__":
    try:
        # input_folder="/data2/TwitterStreamRawData/TwitterData_2020/2020JunCleanedTweets/"
    # outFilepath = 'out/drug_classifier/2020Jun'
        input_folder = sys.argv[1]
        outFilepath = sys.argv[2]
        print(input_folder, outFilepath)
        print(hyperparams)
        predict_for_folder(input_folder, outFilepath)
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
        
        
# usage: python3 Predict_new.py /data2/julina/scripts/tweets/2019/03/csv/ out/drug_classifier/2019/03/
