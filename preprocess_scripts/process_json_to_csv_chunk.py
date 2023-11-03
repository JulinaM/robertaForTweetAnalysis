import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer # tokenize words
from nltk.corpus import stopwords
import pandas as pd
import traceback, os, sys

pd.set_option('display.max_colwidth', None)
# visualization
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams["figure.figsize"] = (10, 8) # default plot size
import seaborn as sns
sns.set(style='whitegrid', palette='Dark2')

def clean_df(filename):
    #df = pd.read_json(filename)
    df = pd.read_json(filename, lines = True)
    print(df.shape)
    #remove duplicate rows
    df = df[['created_at','text']]
    df = df.dropna()
    df['text'] = df['text'].astype(str)
    df.drop_duplicates('text')
    df = df[~(df.text.str.len() < 10)]
    print(df.shape)
    return df

import emoji
import contractions
import re 
import string
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
lmt = WordNetLemmatizer()
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#add your contraction here
contractions.add("cann't", 'can not')


def clean_post(text):
    # lower each word in post
    text = text.lower()
    
    #fix the contractions, for example i'm to i am 
    text = contractions.fix(text)
    # remove multiple spaces
    text = re.sub("\s+"," ",text)
    #change emoji into text
    text = emoji.demojize(text, delimiters=("", ""))
    # remove links
    text = re.sub(r"(http?\://|https?\://|www)\S+", "HTTPURL", text)
    # remove mentions
    text = re.sub("@\w+","@USER",text)
    # alphanumeric and hashtags
    #remove hashtag
    text = re.sub("#[A-Za-z0-9_]+","HASHTAG", text)
  
    tokens = text.split()
    #remove stopwords with custerm additinal words
    additional  = ['rt','rts','retweet','st','n','m','be','amp','bday','offen','iam','want', 'like','cannt']
    stop = set().union(stopwords.words('english'),additional)
    stop_tokens = [t for t in tokens if not t in stop]
    #remove too short capital<2
    clean_tokens = [t for t in stop_tokens if len(t)>2 and len(t)<20]

    #lem = [lmt.lemmatize(t,'v') for t in clean_tokens]
    #lemall = [lmt.lemmatize(t) for t in lem]
    #stem = [stemmer.stem(t) for t in lemall]

    text = " ".join(clean_tokens)
    text = [char for char in text if char not in string.punctuation]
    text = ''.join(text)
    
    return(text)

#founction2 : lematize with post_tag
#https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)

# Function to read the JSON file in chunks
def read_json_in_chunks(file_path, chunk_size=10):
    # print('>'*40, file_path)
    with open(file_path, 'r') as file:
        # print('<'*40, file_path)

        chunk = []
        for line in file:
            try:
                data = json.loads(line)
                chunk.append(data)
            except:
                # Handle invalid JSON, skip or log the error
                # traceback.print_exc()
                # break
                pass
                # continue

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        # print('>'*40, chunk)

        if chunk:
            yield chunk
        

# Function to preprocess each chunk of data
def preprocess_chunk(chunk):
    df = pd.DataFrame(chunk)
    print(df.shape)
    
    #remove duplicate rows
    df = df[['created_at','text']]
    df = df.dropna()
    df['text'] = df['text'].astype(str)
    df.drop_duplicates('text')
    df = df[~(df.text.str.len() < 10)]
    print(df.shape)
    
    df['text'] = df['text'].apply(lambda x: lemmatize_with_postag(clean_post(x)))
    df = df[df['text'].notnull()]
    df = df[~(df.text.str.len() < 10)]
    print(df.shape)
    return df


def text_preprocess(input_folder, output_folder, chunk_size):
    # week_folders = ['2019Week1', '2019Week2', '2019Week3', '2019Week4']
    # for week in week_folders:
        main_path = os.path.join(input_folder)
        for filename in os.listdir(main_path):
            if filename.endswith('.json'):
                filepath = os.path.join(main_path, filename)
                print('>'*40, filepath)
                try:
                    output_filename = output_folder + filename[0:8]+'.csv' 
                    if os.path.exists(output_filename):
                        print(f"The file '{output_filename}' exists.")
                        continue

                    df_n = pd.DataFrame()
                    for chunk in read_json_in_chunks(filepath, chunk_size):
                        chunk_df = preprocess_chunk(chunk)
                        df_n = df_n._append(chunk_df, ignore_index=True)
                    df_n.to_csv(output_filename)
                    print(output_filename, ':: ', df_n.shape)
                    print(f"Done for {filename}!" )
                except:
                    traceback.print_exc()
                    # break
                    pass
    

#usage:: python3 process_json_to_csv_chunk.py /data2/julina/scripts/tweets/2021/10/ /data2/julina/scripts/tweets/2021/10/csv/ 

if __name__ == "__main__":
    try:
        # input_filename = '/data2/julina/scripts/tweets/2019/11/2019_11_01.json'
        # print(input_filename[-10:-5])
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        print(':' * 50, input_folder, output_folder, 1000000)
        text_preprocess(input_folder, output_folder, 1000000)
    except:
        traceback.print_exc()
        print("missing arguments!!!!")
        exit(0)  
        
