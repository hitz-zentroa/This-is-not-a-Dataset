import datasets

from patterns import id_to_pattern
import json


push_to_hub = True

train_json = []
dev_json = []
test_json = []
all_json = []

current_id = 0
current_test = 0

for i in range(1, len(id_to_pattern) + 1):
    print(i)
    train_file = f"This-is-not-a-dataset/en/#{i:02}.train.json"
    dev_file = f"This-is-not-a-dataset/en/#{i:02}.dev.json"
    test_file = f"This-is-not-a-dataset/en/#{i:02}.test.json"

    with open(train_file, "r", encoding="utf8") as f:
        for line in f:
            json_line = json.loads(line)
            json_line["pattern_id"] = i
            json_line["pattern"] = id_to_pattern[i]

            example_id = json_line["test_id"]
            test_no = int(example_id.split("-")[0])
            if test_no < current_id:
                current_test += 1
            current_id = test_no

            json_line["test_id"] = current_test

            # Make patter_id and pattern the first two keys
            json_line = {
                k: json_line[k]
                for k in ["pattern_id", "pattern", "test_id"] + list(json_line.keys())
                if k in json_line
            }
            train_json.append(json_line)

    with open(dev_file, "r", encoding="utf8") as f:
        for line in f:
            json_line = json.loads(line)
            json_line["pattern_id"] = i
            json_line["pattern"] = id_to_pattern[i]

            example_id = json_line["test_id"]
            test_no = int(example_id.split("-")[0])
            if test_no < current_id:
                current_test += 1
            current_id = test_no

            json_line["test_id"] = current_test

            # Make patter_id and pattern the first two keys
            json_line = {
                k: json_line[k]
                for k in ["pattern_id", "pattern", "test_id"] + list(json_line.keys())
                if k in json_line
            }
            dev_json.append(json_line)

    with open(test_file, "r", encoding="utf8") as f:
        for line in f:
            json_line = json.loads(line)
            json_line["pattern_id"] = i
            json_line["pattern"] = id_to_pattern[i]

            example_id = json_line["test_id"]
            test_no = int(example_id.split("-")[0])
            if test_no < current_id:
                current_test += 1
            current_id = test_no

            json_line["test_id"] = current_test

            # Make patter_id and pattern the first two keys
            json_line = {
                k: json_line[k]
                for k in ["pattern_id", "pattern", "test_id"] + list(json_line.keys())
                if k in json_line
            }
            test_json.append(json_line)

for i in range(1, len(id_to_pattern) + 1):
    print(i)
    all_file = f"This-is-not-a-dataset/en/#{i:02}.all.json"

    with open(all_file, "r", encoding="utf8") as f:
        for line in f:
            json_line = json.loads(line)
            json_line["pattern_id"] = i
            json_line["pattern"] = id_to_pattern[i]

            example_id = json_line["test_id"]
            test_no = int(example_id.split("-")[0])
            if test_no < current_id:
                current_test += 1
            current_id = test_no

            json_line["test_id"] = current_test

            # Make patter_id and pattern the first two keys
            json_line = {
                k: json_line[k]
                for k in ["pattern_id", "pattern", "test_id"] + list(json_line.keys())
                if k in json_line
            }
            all_json.append(json_line)

with open("This-is-not-a-dataset/en/train.jsonl", "w", encoding="utf8") as f:
    for line in train_json:
        print(json.dumps(line, ensure_ascii=False), file=f)

with open("This-is-not-a-dataset/en/dev.jsonl", "w", encoding="utf8") as f:
    for line in dev_json:
        print(json.dumps(line, ensure_ascii=False), file=f)

with open("This-is-not-a-dataset/en/test.jsonl", "w", encoding="utf8") as f:
    for line in test_json:
        print(json.dumps(line, ensure_ascii=False), file=f)

with open("This-is-not-a-dataset/en/all.jsonl", "w", encoding="utf8") as f:
    for line in all_json:
        print(json.dumps(line, ensure_ascii=False), file=f)


dataset = datasets.load_dataset(
    path="This-is-not-a-dataset/en",
    data_files={
        "train": "train.jsonl",
        "validation": "dev.jsonl",
        "test": "test.jsonl",
    },
)

#  ['pattern_id', 'pattern', 'test_id', 'negation_type', 'semantic_type', 'syntactic_scope', 'isDistractor', 'label', 'sentence'],
dataset_info = datasets.DatasetInfo(
    description="We introduce a large semi-automatically generated dataset of ~400,000 descriptive sentences about commonsense knowledge that can be true or false in which negation is present in about 2/3 of the corpus in different forms that we use to evaluate LLMs",
    features=datasets.Features(
        {
            "pattern_id": datasets.Value("int32"),
            "pattern": datasets.Value("string"),
            "test_id": datasets.Value("int32"),
            "negation_type": datasets.Value("string"),
            "semantic_type": datasets.Value("string"),
            "syntactic_scope": datasets.Value("string"),
            "isDistractor": datasets.Value("bool"),
            "label": datasets.Value("bool"),
            "sentence": datasets.Value("string"),
        }
    ),
    supervised_keys=datasets.info.SupervisedKeysData(input="sentence", output="label"),
    homepage="https://github.com/hitz-zentroa/This-is-not-a-Dataset",
    citation="""@inproceedings{this-is-not-a-dataset,
    title = "This is not a Dataset: A Large Negation Benchmark to Challenge Large Language Models",
    author = "Iker García-Ferrero, Begoña Altuna, Javier Alvez, Itziar Gonzalez-Dios, German Rigau",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    }
    """,
)
dataset.features = dataset_info.features
dataset.supervised_keys = dataset_info.supervised_keys
dataset.description = dataset_info.description
dataset.citation = dataset_info.citation
dataset.homepage = dataset_info.homepage
print(dataset)
print(dataset["train"][0])
print(dataset["validation"][0])
print(dataset["test"][0])
print(dataset.features)
print(dataset.description)
print(dataset.citation)
print(dataset.homepage)
print(dataset.supervised_keys)

if push_to_hub:
    print("Pushing to hub...")
    dataset.push_to_hub("HiTZ/This-is-not-a-dataset")
    print("Done!")
