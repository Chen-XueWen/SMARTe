import argparse
from transformers import AutoTokenizer
import pickle
import json
from tqdm import tqdm
from utils import remove_accents, list_index

#This preprocessing code is adapted from the SPN4RE repository: https://github.com/DianboWork/SPN4RE.

def read_data(file_in, tokenizer, rs_mapping):
    samples = []
    gold_num = 0

    with open(file_in) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]
        
    for i in tqdm(range(len(lines))):
        token_sent = [tokenizer.cls_token] + tokenizer.tokenize(remove_accents(lines[i]["sentText"])) + [tokenizer.sep_token]
        
        triples = lines[i]["relationMentions"]
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": []}
        gold_num += len(triples)
        for triple in triples:
            head_entity = remove_accents(triple["em1Text"])
            tail_entity = remove_accents(triple["em2Text"])
            head_token = tokenizer.tokenize(head_entity)
            tail_token = tokenizer.tokenize(tail_entity)
            try:
                relation_id = rs_mapping[triple["label"]]
            except:
                print(f'Relation \"{triple["label"]}\" is missing in training, removing from validation dataset')
                break
                
            head_start_index, head_end_index = list_index(head_token, token_sent)
            assert head_end_index >= head_start_index
            tail_start_index, tail_end_index = list_index(tail_token, token_sent)
            assert tail_end_index >= tail_start_index
            target["relation"].append(relation_id)
            target["head_start_index"].append(head_start_index)
            target["head_end_index"].append(head_end_index)
            target["tail_start_index"].append(tail_start_index)
            target["tail_end_index"].append(tail_end_index)
        sent_id = tokenizer.convert_tokens_to_ids(token_sent)
        samples.append([i, sent_id, target])
    
    return samples

def main(data_type):
    
    print("Processing Data:", data_type)
    
    train_file = f"./data/{data_type}/train.json"
    dev_file = f"./data/{data_type}/valid.json"
    test_file = f"./data/{data_type}/test.json"

    train_out = f"./processed/{data_type}/train_features.pkl"
    dev_out = f"./processed/{data_type}/dev_features.pkl"
    test_out = f"./processed/{data_type}/test_features.pkl"
    traindev_out = f"./processed/{data_type}/traindev_features.pkl"
    
    rs_mapping = json.load(open(f"./data/{data_type}/map.json"))
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    train_features = read_data(train_file, tokenizer, rs_mapping)
    dev_features = read_data(dev_file, tokenizer, rs_mapping)
    test_features = read_data(test_file, tokenizer, rs_mapping)

    traindev_features = train_features + dev_features
    
    outfile = open(train_out, 'wb')
    pickle.dump(train_features, outfile)
    outfile.close()

    outfile = open(dev_out, 'wb')
    pickle.dump(dev_features, outfile)
    outfile.close()

    outfile = open(test_out, 'wb')
    pickle.dump(test_features, outfile)
    outfile.close()
    
    outfile = open(traindev_out, 'wb')
    pickle.dump(traindev_features, outfile)
    outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw dataset for relation extraction")
    parser.add_argument("--data_type", type=str, required=True, help="Dataset type (e.g., NYT-Exact)")
    
    args = parser.parse_args()
    main(args.data_type)