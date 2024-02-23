import json

from patterns import id_to_pattern, pattern_to_id


def get_few_shot():
    examples_dict = {
        "Synonymy1": [],
        "Antonymy1": [],
        "Synonymy2": [],
        "Antonymy2": [],
        "Hypernymy": [],
        "Part": [],
        "Substance": [],
        "Member": [],
        "Agent": [],
        "Instrument": [],
        "Result": [],
    }

    # Open each file. Get the first examples that:
    # isDistractor == False, Label = True
    # isDistractor == False, Label = False
    # isDistractor == True, Label = True
    # isDistractor == True, Label = False

    for i, pattern_name in id_to_pattern.items():
        with open(f"This-is-not-a-dataset/en/#{i:02}.train.json", "r") as f:
            for line in f:
                example = json.loads(line)
                if not example["isDistractor"] and example["label"]:
                    examples_dict[pattern_name].append(example)
                    break
            for line in f:
                example = json.loads(line)
                if not example["isDistractor"] and not example["label"]:
                    examples_dict[pattern_name].append(example)
                    break
            for line in f:
                example = json.loads(line)
                if example["isDistractor"] and example["label"]:
                    examples_dict[pattern_name].append(example)
                    break
            for line in f:
                example = json.loads(line)
                if example["isDistractor"] and not example["label"]:
                    examples_dict[pattern_name].append(example)
                    break

    # Convert examples_dict to a list of examples
    examples = []
    for pattern_name, pattern_examples in examples_dict.items():
        for example in pattern_examples:
            example["pattern"] = pattern_to_id[pattern_name]
            examples.append(example)

    print(
        "\n".join(
            example["sentence"] + " " + str(example["label"]) for example in examples
        )
    )


if __name__ == "__main__":
    get_few_shot()
