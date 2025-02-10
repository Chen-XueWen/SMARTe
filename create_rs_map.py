import json
import argparse
import os

def main(dataset_name):
    train_file = f"./data/{dataset_name}/train.json"
    train_out = f"./data/{dataset_name}/map.json"

    if not os.path.exists(train_file):
        print(f"Error: {train_file} does not exist.")
        return

    with open(train_file, "r") as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]

    relations = []
    relations_dict = {}
    relations_idx = 0

    for i in range(len(lines)):
        triples = lines[i]["relationMentions"]
        for triple in triples:
            relations.append(triple["label"])
            if triple["label"] not in relations_dict:
                relations_dict[triple["label"]] = relations_idx
                relations_idx += 1

    with open(train_out, "w") as f:
        json.dump(relations_dict, f, indent=4)

    print(f"Relation mapping saved to {train_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get all relationship in train dataset and create a map")
    parser.add_argument("--data_type", type=str, required=True, help="Name of the dataset folder (e.g., NYT-Exact)")

    args = parser.parse_args()
    main(args.data_type)