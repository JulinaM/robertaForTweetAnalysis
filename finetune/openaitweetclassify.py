
import os
import openai
import numpy as np
import pandas as pd
import re
import json
from datetime import datetime
import time
import random

#https://platform.openai.com/docs/guides/error-codes
#https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def cleaner(tweet):
        tweet = re.sub("@[A-Za-z0-9]+","", tweet) #Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
        tweet = " ".join(tweet.split())
        #     tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
        #     tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        #     tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
        #          if w.lower() in words or not w.isalpha())
       
        return tweet


openai.api_key = "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"
# openai.api_key = "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"

df_pred=pd.read_csv('/users/kent/jmaharja/drugAbuse/finetune/reviewed_test_result/2023_neg.csv',lineterminator='\n',skipinitialspace=True,)
df_pred = df_pred.rename(columns={df_pred.columns[1]: 'Tweet'})
print(df_pred.shape)
df_pred = df_pred[6000:10000]
print(df_pred.shape)
df_pred['Tweet'] = df_pred['Tweet'].map(lambda x: cleaner(x))



# df1 =pd.read_csv('/users/kent/jmaharja/drugAbuse/input/Tweets_Spring_Summer_2021_coded.csv',lineterminator='\n',skipinitialspace=True,)
# df1_pos = df1.loc[(df1['Substance'] != 'X') & (df1['Use'] != 'X') & (df1['Intent'] != 'X')]
# df1_neg = df1.loc[(df1['Substance'] == 'X') & (df1['Use'] == 'X') & (df1['Intent'] == 'X')]
# df1_pos.drop(df1_pos.columns[[ 2,3,4]], axis=1, inplace=True)
# df1_neg.drop(df1_neg.columns[[ 2,3,4]], axis=1, inplace=True)
# df1_pos.rename(columns = {'Number':'id'}, inplace = True)
# df1_neg.rename(columns = {'Number':'id'}, inplace = True)
# # df1_neg = df1_neg[:100]
# # df1_pos = df1_pos[:100]
# df1_pos['label']= 1
# df1_neg['label']= 0

# df_pred=df1_pos
# df_pred['Tweet'] = df_pred['Tweet'].map(lambda x: cleaner(x))
# print(df_pred.shape)
        
cur_t =datetime.now().strftime("%Y_%m_%d-%I_%M%p")
res = pd.DataFrame(columns = ['id', 'Tweet', 'robert_label', 'human_label', 'gpt_label'])
i =0
for index, row in df_pred.iterrows():
    print(i)
    tweet = row['Tweet']
    response = completions_with_backoff(
#     response = openai.Completion.create(
        model="text-davinci-003",
        prompt ="Is this tweet" + tweet + "related to drug abuse: Yes or No?"  ,
        temperature=0.5,
        max_tokens=144,
        top_p=0.6,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    res = pd.concat([res, pd.DataFrame.from_records([{'id': row['id'], 
                                                      'Tweet' : tweet,
                                                      'robert_label' : row['robert_label'],
                                                      'human_label' : row['human_label'],
                                                      'gpt_label' : response['choices'][0]["text"].strip(),
                                                     }])], ignore_index=True)
    i = i+1



print(res)
res.to_csv('reviewed_test_result/2023_neg_gpt_10k.csv')
    
