# from flask import Flask
from flask import Flask, request
from transformers import RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import transformers
from transformers import BertModel, BertTokenizer
from transformers import RobertaConfig
from torch import nn
import torch
import json

MAX_LEN = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer_folder = '/data/jmharja/projects/robertaForTweetAnalysis/oct2022/TokRoBERTa'
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=MAX_LEN)

class TweetModel(RobertaPreTrainedModel):
    def __init__(self, conf, n_classes):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained('/data/jmharja/projects/robertaForTweetAnalysis/oct2022/RoBERTaMLM/', config=conf)
        self.drop_out = nn.Dropout(0.5)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.classifier = nn.Linear(768, n_classes)
        
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
model.load_state_dict(torch.load('/data/jmharja/projects/robertaForTweetAnalysis/finetune/checkpoint/best_ftc_model_state2023_02_20-03_15PM.bin'))


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get request data
        data = request.json
        tweet= data["tweet"]
#         tweet = "Some shots from my shoot for F.I.T. Studio. Coach Fred"
        encoded_tweet = tokenizer.encode_plus(
            tweet,
            max_length=MAX_LEN,
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        input_ids = encoded_tweet['input_ids'].to(device)
        attention_mask = encoded_tweet['attention_mask'].to(device)
        token_type_ids = encoded_tweet['token_type_ids'].to(device)

        output = model(input_ids, attention_mask, token_type_ids)
        _, prediction = torch.max(output, dim=1)
        print(f'Tweet text: {tweet}')
        print(f'Substance type  : {prediction}')
#         return jsonify({'result': result})
        tensor_json = json.dumps(prediction.tolist())
        return tweet + "--->" + tensor_json

       
    
    
# @app.route('/')
# def hello():
#     return "Hello World!"