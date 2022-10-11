import json
import pandas as pd
import sys


def do(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data, columns=['id', 'created_at', 'text','lang'])  # change the feature in the columns from the original Twitter structure
    print(df.shape)
    # write to a new csv file
#     df.to_csv('./tweets_valid.csv')


if __name__ == "__main__":
    # filename = './tweets_valid.csv'

    filename = sys.argv[1]
    do("temp/test_scripts/json2csv.py tweets_valid.json")
