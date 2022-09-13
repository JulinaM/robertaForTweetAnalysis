import os

from transformers import pipeline

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

fill_mask = pipeline(
    "fill-mask",
    model=model_folder,
    tokenizer=tokenizer_folder
)

fill_mask("Alcohol and drugs is good for the <mask>")
