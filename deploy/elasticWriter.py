from pymongo import MongoClient
from elasticsearch import Elasticsearch
from bson import json_util  
#import bson.json_util as bson_json
import json
import bson
import requests
# MongoDB connection settings
mongo_host = 'localhost'
mongo_port = 27017
mongo_db = 'tweet_drugs'
mongo_collection = "tweets_valid"

# Elasticsearch connection settings
es_host = 'localhost'
es_port = 9200
es_index = "test_july11_v1"
es_doc_type = "tweets_valid"
ELASTIC_PASSWORD = "UkojPRKEESb5z3Z=amy-"

def convert_bson_to_json(bson_doc, old_field_name, new_field_name):
    # Decode BSON to Python dictionary
    #doc_dict = bson.decode_all(bson_doc)
    doc_dict = bson_json.loads(bson_doc)

    # Rename the field
    for doc in doc_dict:
        if old_field_name in doc:
            doc[new_field_name] = doc.pop(old_field_name)

    # Convert the dictionary to JSON
    json_doc = json.dumps(doc_dict)
    return json_doc


# Connect to MongoDB
mongo_client = MongoClient(mongo_host, mongo_port)
mongo_db = mongo_client[mongo_db]
mongo_collection = mongo_db[mongo_collection]

# Connect to Elasticsearch
#es = Elasticsearch([{"host": es_host, "port": es_port}])
# Create the client instance
es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="/opt/elasticstack/elasticsearch-8.6.2/config/certs/http_ca.crt",
    basic_auth=("elastic", ELASTIC_PASSWORD)
)


mongo_data = mongo_collection.find(no_cursor_timeout=True)
#batch_size = 1000
#for index in range(0, mongo_data.count(), batch_size):
#    print('------- ',index, batch_size, mongo_data.count())
#    batch = mongo_data.skip(index).limit(batch_size)
#    for document in batch:
#        # Process each document
#        json_str = json_util.dumps(document)
#        doc = json.loads(json_str)
#        doc["mongo_id"] = doc.pop('_id')
#        res = es.index(index=es_index, document=doc)

# Iterate through MongoDB documents
for document in mongo_data:
    #if document['id'] == "1000005666982023168": 
    # Write the modified document to Elasticsearch
    json_str = json_util.dumps(document)
    doc = json.loads(json_str)
    doc['mongo_id'] = doc.pop('_id')
#     print(doc['text'])
    query = {"tweet": doc['text']}
    headers =  {"Content-Type":"application/json"}
#     response = requests.post("http://localhost:5000/", data=json.dumps(query), headers= headers)
    response = requests.post("http://localhost:5000/", params=query, headers= headers)
#     print(response.text)
    res = json.loads(response.text)
    doc['pred'] = res['pred']
    res = es.index(index=es_index, document=doc)
#     print(doc['mongo_id'])
mongo_data.close()
mongo_client.close()