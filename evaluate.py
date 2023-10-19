import json
from typing import Dict, List, Any, Optional
import argparse


def convert_to_bool(value):
    if isinstance(value, str) and value.lower().strip() in ["true", "false"]:
        value = value.lower().strip() == "true"
    elif isinstance(value, bool):
        pass
    elif isinstance(value, int) and value in [0, 1]:
        value = value == 1
    elif isinstance(value, float) and int(value) in [
        0,
    ]:
        value = int(value) == 1
    else:
        raise Exception(f"Unknown value type: {type(value)}")

    return value


class Scorer:
    def __init__(self):
        self.results = {}

    def add_example(
        self,
        negation_type,
        semantic_type,
        syntactic_scope,
        isDistractor,
        gold_label,
        predicted_label,
    ):
        gold_label = convert_to_bool(gold_label)
        predicted_label = convert_to_bool(predicted_label)
        isDistractor = convert_to_bool(isDistractor)

        # Compute score affirmation/negation
        if negation_type == "affirmation":
            if "all_affirmations" not in self.results:
                self.results["all_affirmations"] = {
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "TN": 0,
                }
            if gold_label == predicted_label:
                if gold_label:
                    self.results["all_affirmations"]["TP"] += 1
                else:
                    self.results["all_affirmations"]["TN"] += 1
            else:
                if gold_label:
                    self.results["all_affirmations"]["FN"] += 1
                else:
                    self.results["all_affirmations"]["FP"] += 1

        else:
            if "all_negations" not in self.results:
                self.results["all_negations"] = {
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "TN": 0,
                }
            if gold_label == predicted_label:
                if gold_label:
                    self.results["all_negations"]["TP"] += 1
                else:
                    self.results["all_negations"]["TN"] += 1
            else:
                if gold_label:
                    self.results["all_negations"]["FN"] += 1
                else:
                    self.results["all_negations"]["FP"] += 1

        # Compute distractor scores
        if isDistractor and negation_type == "affirmation":
            key = "distractor_affirmation"
        elif isDistractor and negation_type != "affirmation":
            key = "distractor_negation"
        elif not isDistractor and negation_type == "affirmation":
            key = "input_affirmation"
        else:
            key = "input_negation"
        if key not in self.results:
            self.results[key] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        if gold_label == predicted_label:
            if gold_label:
                self.results[key]["TP"] += 1
            else:
                self.results[key]["TN"] += 1
        else:
            if gold_label:
                self.results[key]["FN"] += 1
            else:
                self.results[key]["FP"] += 1

        # Compute overall scores

        if "all" not in self.results:
            self.results["all"] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        if gold_label == predicted_label:
            if gold_label:
                self.results["all"]["TP"] += 1
            else:
                self.results["all"]["TN"] += 1
        else:
            if gold_label:
                self.results["all"]["FN"] += 1
            else:
                self.results["all"]["FP"] += 1

        ### Compute score each negation type

        if "Negation_analysis" not in self.results:
            self.results["Negation_analysis"] = {}

        for negation_type in [
            negation_type.strip().lower(),
            semantic_type.strip().lower(),
            syntactic_scope.strip().lower(),
        ]:
            if negation_type != "affirmation" and negation_type != "none":
                if f"{negation_type}_all" not in self.results["Negation_analysis"]:
                    self.results["Negation_analysis"][f"{negation_type}_all"] = {
                        "TP": 0,
                        "FP": 0,
                        "FN": 0,
                        "TN": 0,
                    }
                    self.results["Negation_analysis"][f"{negation_type}_input"] = {
                        "TP": 0,
                        "FP": 0,
                        "FN": 0,
                        "TN": 0,
                    }
                    self.results["Negation_analysis"][f"{negation_type}_distractor"] = {
                        "TP": 0,
                        "FP": 0,
                        "FN": 0,
                        "TN": 0,
                    }

                if gold_label == predicted_label:
                    if gold_label:
                        self.results["Negation_analysis"][f"{negation_type}_all"][
                            "TP"
                        ] += 1
                        if isDistractor:
                            self.results["Negation_analysis"][
                                f"{negation_type}_distractor"
                            ]["TP"] += 1
                        else:
                            self.results["Negation_analysis"][f"{negation_type}_input"][
                                "TP"
                            ] += 1
                    else:
                        self.results["Negation_analysis"][f"{negation_type}_all"][
                            "TN"
                        ] += 1
                        if isDistractor:
                            self.results["Negation_analysis"][
                                f"{negation_type}_distractor"
                            ]["TN"] += 1
                        else:
                            self.results["Negation_analysis"][f"{negation_type}_input"][
                                "TN"
                            ] += 1
                else:
                    if gold_label:
                        self.results["Negation_analysis"][f"{negation_type}_all"][
                            "FN"
                        ] += 1
                        if isDistractor:
                            self.results["Negation_analysis"][
                                f"{negation_type}_distractor"
                            ]["FN"] += 1
                        else:
                            self.results["Negation_analysis"][f"{negation_type}_input"][
                                "FN"
                            ] += 1
                    else:
                        self.results["Negation_analysis"][f"{negation_type}_all"][
                            "FP"
                        ] += 1
                        if isDistractor:
                            self.results["Negation_analysis"][
                                f"{negation_type}_distractor"
                            ]["FP"] += 1
                        else:
                            self.results["Negation_analysis"][f"{negation_type}_input"][
                                "FP"
                            ] += 1

    def compute_scores(self):
        for key in self.results:
            if key != "Negation_analysis":
                TP = self.results[key]["TP"]
                FP = self.results[key]["FP"]
                FN = self.results[key]["FN"]
                TN = self.results[key]["TN"]
                self.results[key]["accuracy"] = (
                    round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)
                    if (TP + TN + FP + FN) > 0
                    else 0
                )
        if "Negation_analysis" in self.results:
            for key in self.results["Negation_analysis"]:
                TP = self.results["Negation_analysis"][key]["TP"]
                FP = self.results["Negation_analysis"][key]["FP"]
                FN = self.results["Negation_analysis"][key]["FN"]
                TN = self.results["Negation_analysis"][key]["TN"]
                self.results["Negation_analysis"][key]["accuracy"] = (
                    round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)
                    if (TP + TN + FP + FN) > 0
                    else 0
                )

            # Ensure that Negation_analysis is the last key in the results dictionary for better readability
            negation_analysis = self.results.pop("Negation_analysis")
            self.results["Negation_analysis"] = negation_analysis

        return self.results


class Coherence_Scorer:
    def __init__(self):
        self.test_results = {
            "All": {"Coherent": 0, "Incoherent": 0},
            "Affirmation-Negation_Input": {"Coherent": 0, "Incoherent": 0},
            "Affirmation-Negation_Distractor": {"Coherent": 0, "Incoherent": 0},
            "Input-Distractor_Affirmation": {"Coherent": 0, "Incoherent": 0},
            "Input-Distractor_Negation": {"Coherent": 0, "Incoherent": 0},
        }

    def add_test(
        self,
        examples: List[Dict[str, Any]],
        test_no: int,
    ):
        # print(len(examples))
        affirmative_answers = []
        negated_answers = []
        affirmative_distractor_answers = []
        negated_distractor_answers = []
        for example in examples:
            example["label"] = convert_to_bool(example["label"])
            example["prediction"] = convert_to_bool(example["prediction"])

            isDistractor = convert_to_bool(example["isDistractor"])
            negation_type = example["negation_type"]
            if test_no != 2 and test_no != 4:
                if (
                    isDistractor
                    and negation_type == "affirmation"
                    and example["label"] is True
                ):
                    continue
                elif (
                    isDistractor
                    and negation_type != "affirmation"
                    and example["label"] is False
                ):
                    continue
                elif (
                    not isDistractor
                    and negation_type == "affirmation"
                    and example["label"] is False
                ):
                    continue
                elif (
                    not isDistractor
                    and negation_type != "affirmation"
                    and example["label"] is True
                ):
                    continue
            else:
                if (
                    isDistractor
                    and negation_type == "affirmation"
                    and example["label"] is True
                ):
                    continue
                elif (
                    isDistractor
                    and negation_type != "affirmation"
                    and example["label"] is False
                ):
                    continue
                elif (
                    not isDistractor
                    and negation_type == "affirmation"
                    and example["label"] is True
                ):
                    continue
                elif (
                    not isDistractor
                    and negation_type != "affirmation"
                    and example["label"] is False
                ):
                    continue

            if isDistractor and negation_type == "affirmation":
                affirmative_distractor_answers.append(example["prediction"])
            elif isDistractor and negation_type != "affirmation":
                negated_distractor_answers.append(example["prediction"])
            elif not isDistractor and negation_type == "affirmation":
                affirmative_answers.append(example["prediction"])
            else:
                negated_answers.append(example["prediction"])

        non_distractor_coherent = True
        distractor_coherent = True

        # print(f"Affirmative Answers: {affirmative_answers}")
        # print(f"Negated Answers: {negated_answers}")
        # print(f"Affirmative Distractor Answers: {affirmative_distractor_answers}")
        # print(f"Negated Distractor Answers: {negated_distractor_answers}")

        # Affirmation-Negation_Input
        # If all answers are not the same, then the answer is incoherent
        if len(set(affirmative_answers)) > 1 or len(set(negated_answers)) > 1:
            self.test_results["Affirmation-Negation_Input"]["Incoherent"] += 1
            non_distractor_coherent = False

        else:
            # If all in affirmative_answers is True, then all negated_answers should be False and vice versa
            if len(affirmative_answers) == 0 or len(negated_answers) == 0:
                print(json.dumps(examples, indent=4))
            if affirmative_answers[0] == negated_answers[0]:
                self.test_results["Affirmation-Negation_Input"]["Incoherent"] += 1
                non_distractor_coherent = False
            else:
                self.test_results["Affirmation-Negation_Input"]["Coherent"] += 1

        # Compute Affirmation-Negation_Distractor
        # If all answers are not the same, then the answer is incoherent
        if (
            len(set(affirmative_distractor_answers)) > 1
            or len(set(negated_distractor_answers)) > 1
        ):
            self.test_results["Affirmation-Negation_Distractor"]["Incoherent"] += 1
            distractor_coherent = False

        else:
            # If all in affirmative_answers is True, then all negated_answers should be False and vice versa
            if affirmative_distractor_answers[0] == negated_distractor_answers[0]:
                self.test_results["Affirmation-Negation_Distractor"]["Incoherent"] += 1
                distractor_coherent = False
            else:
                self.test_results["Affirmation-Negation_Distractor"]["Coherent"] += 1

        # Compute overall coherence
        # Requeriments:
        # 1. Non-distractor answers should be coherent
        # 2. Distractor answers should be coherent
        # 3. All the answers should be correct or all the answers should be incorrect

        all_correct = True
        all_incorrect = True
        for example in examples:
            if example["label"] == example["prediction"]:
                all_incorrect = False
            else:
                all_correct = False

        if (
            non_distractor_coherent
            and distractor_coherent
            and (all_correct or all_incorrect)
        ):
            self.test_results["All"]["Coherent"] += 1
        else:
            self.test_results["All"]["Incoherent"] += 1

        if test_no != 2 and test_no != 4:
            # Compute Input-Distractor_Affirmation
            if (
                len(set(affirmative_answers)) > 1
                or len(set(affirmative_distractor_answers)) > 1
            ):
                self.test_results["Input-Distractor_Affirmation"]["Incoherent"] += 1
            else:
                if affirmative_answers[0] == affirmative_distractor_answers[0]:
                    self.test_results["Input-Distractor_Affirmation"]["Incoherent"] += 1
                else:
                    self.test_results["Input-Distractor_Affirmation"]["Coherent"] += 1

            # Compute Input-Distractor_Negation
            if (
                len(set(negated_answers)) > 1
                or len(set(negated_distractor_answers)) > 1
            ):
                self.test_results["Input-Distractor_Negation"]["Incoherent"] += 1
            else:
                if negated_answers[0] == negated_distractor_answers[0]:
                    self.test_results["Input-Distractor_Negation"]["Incoherent"] += 1
                else:
                    self.test_results["Input-Distractor_Negation"]["Coherent"] += 1
        else:
            # For test 2 and 4, the sentence w/wo distractor are both True or both False

            # Compute Input-Distractor_Affirmation
            if (
                len(set(affirmative_answers)) > 1
                or len(set(affirmative_distractor_answers)) > 1
            ):
                self.test_results["Input-Distractor_Affirmation"]["Incoherent"] += 1
            else:
                if affirmative_answers[0] != affirmative_distractor_answers[0]:
                    self.test_results["Input-Distractor_Affirmation"]["Incoherent"] += 1
                else:
                    self.test_results["Input-Distractor_Affirmation"]["Coherent"] += 1
            # Compute Input-Distractor_Negation
            if (
                len(set(negated_answers)) > 1
                or len(set(negated_distractor_answers)) > 1
            ):
                self.test_results["Input-Distractor_Negation"]["Incoherent"] += 1
            else:
                if negated_answers[0] != negated_distractor_answers[0]:
                    self.test_results["Input-Distractor_Negation"]["Incoherent"] += 1
                else:
                    self.test_results["Input-Distractor_Negation"]["Coherent"] += 1

        # print(
        #    f"affirmative_answers: {affirmative_answers} "
        #    f"negated_answers: {negated_answers} "
        #    f"affirmative_distractor_answers: "
        #    f"{affirmative_distractor_answers} "
        #    f"negated_distractor_answers: {negated_distractor_answers} \n"
        #    f"non_distractor_coherent: {non_distractor_coherent} "
        #    f"distractor_coherent: {distractor_coherent} \n"
        # )

    def compute_scores(self):
        for key in self.test_results:
            coherent_no = self.test_results[key]["Coherent"]
            incoherent_no = self.test_results[key]["Incoherent"]
            self.test_results[key]["Coherence_percentage"] = (
                round((coherent_no / (coherent_no + incoherent_no)) * 100, 2)
                if (coherent_no + incoherent_no) > 0
                else 0
            )
        return self.test_results

    # Create class from file
    @classmethod
    def from_file(cls, file_path: str):
        scorer = cls()

        test_id_dict = {}

        with open(file_path, "r", encoding="utf8") as file:
            for line in file:
                example = json.loads(line.strip())
                example_id = example["test_id"]
                if example_id not in test_id_dict:
                    test_id_dict[example_id] = []

                test_id_dict[example_id].append(example)

        for examples in test_id_dict.values():
            scorer.add_test(examples, test_no=examples[0]["pattern_id"])

        return scorer

    def add_file(self, file_path: str):
        test_id_dict = {}

        with open(file_path, "r", encoding="utf8") as file:
            for line in file:
                example = json.loads(line.strip())
                example_id = example["test_id"]
                if example_id not in test_id_dict:
                    test_id_dict[example_id] = []

                test_id_dict[example_id].append(example)

        for examples in test_id_dict.values():
            self.add_test(examples, test_no=examples[0]["pattern_id"])

    def add_pattern(self, examples: List[Dict[str, Any]]):
        test_id_dict = {}

        for example in examples:
            example_id = example["test_id"]
            if example_id not in test_id_dict:
                test_id_dict[example_id] = []

            test_id_dict[example_id].append(example)

        for test_examples in test_id_dict.values():
            self.add_test(test_examples, test_no=test_examples[0]["pattern_id"])


def evaluate(predictions_path: str, output_path: Optional[str] = None) -> dict:
    """
    Evaluate the predictions of a model
    Args:
        predictions_path: Path to the predictions file. It should be a jsonl with the fields: 'pattern_id',
        'pattern', 'test_id', 'negation_type', 'semantic_type', 'syntactic_scope', 'isDistractor',
        'label', 'sentence', 'prediction'
        output_path: Path to the output file. If None, the output will be printed to stdout
    Returns:
        A dictionary with the scores
        The scorer will output the following metrics:
            - **all_affirmations**: Accuracy of the model in affirmative sentences
            - **all_negations**: Accuracy of the model in negated sentences
            - **all**: (Overall) Accuracy of the model in all sentences
            - **input_affirmation**: Accuracy of the model in affirmative sentences without distractors
            - **input_negation**: Accuracy of the model in negated sentences without distractors
            - **distractor_affirmation**: Accuracy of the model in affirmative sentences with distractors
            - **distractor_negation**: Accuracy of the model in negated sentences with distractors
            - **Negation_analysis**: Fine-grained analysis of the model in negated sentences (verbal, analytic,
            clausal, non_verbal, synthetic, subclausal negation types)
            - **Synonymy1, Hypernymy, Part...**: Fine-grained analysis of the model in each pattern
    """

    print(
        """
*************************************** Running evaluation ***************************************
The scorer will output the following metrics:
    - **all_affirmations**: Accuracy of the model in affirmative sentences
    - **all_negations**: Accuracy of the model in negated sentences
    - **all**: (Overall) Accuracy of the model in all sentences
    - **input_affirmation**: Accuracy of the model in affirmative sentences without distractors
    - **input_negation**: Accuracy of the model in negated sentences without distractors
    - **distractor_affirmation**: Accuracy of the model in affirmative sentences with distractors
    - **distractor_negation**: Accuracy of the model in negated sentences with distractors
    - **Negation_analysis**: Fine-grained analysis of the model in negated sentences (verbal, analytic,
    clausal, non_verbal, synthetic, subclausal negation types)
    - **Synonymy1, Hypernymy, Part...**: Fine-grained analysis of the model in each pattern
**************************************************************************************************
    """
    )
    dataset_pattern = {
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

    scorer = Scorer()
    coherence_scorer = Coherence_Scorer()

    coherence_scorer.from_file(predictions_path)
    with open(predictions_path, "r", encoding="utf8") as file:
        for line in file:
            example = json.loads(line.strip())
            pattern = example["pattern"]
            dataset_pattern[pattern].append(example)
            scorer.add_example(
                negation_type=example["negation_type"],
                semantic_type=example["semantic_type"],
                syntactic_scope=example["syntactic_scope"],
                isDistractor=example["isDistractor"],
                gold_label=example["label"],
                predicted_label=example["prediction"],
            )

    scores = scorer.compute_scores()
    coherence_scorer = Coherence_Scorer.from_file(predictions_path)
    scores["coherence_scores"] = coherence_scorer.compute_scores()

    for pattern in dataset_pattern:
        scorer = Scorer()
        coherence_scorer = Coherence_Scorer()
        coherence_scorer.add_pattern(dataset_pattern[pattern])
        for example in dataset_pattern[pattern]:
            scorer.add_example(
                negation_type=example["negation_type"],
                semantic_type=example["semantic_type"],
                syntactic_scope=example["syntactic_scope"],
                isDistractor=example["isDistractor"],
                gold_label=example["label"],
                predicted_label=example["prediction"],
            )
        scores[pattern] = scorer.compute_scores()
        scores[pattern]["coherence_scores"] = coherence_scorer.compute_scores()

    if output_path is not None:
        print(f"Saving scores to {output_path}")
        with open(output_path, "w", encoding="utf8") as file:
            print(json.dumps(scores, ensure_ascii=False, indent=4), file=file)
    else:
        print(json.dumps(scores, ensure_ascii=False, indent=4))

    print("*** Evaluation finished ***")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # predictions_path, reference_json, output_path
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to the model predictions",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="Path to the output file",
    )

    args = parser.parse_args()

    evaluate(
        predictions_path=args.predictions_path,
        output_path=args.output_path,
    )

"""
python3 evaluate.py \
--predictions_path /ikerlariak/igarcia945/NegationLM/results/zero-shot/vicuna13B/results.json \
--output_path /ikerlariak/igarcia945/NegationLM/results/zero-shot/vicuna13B/scores.json
"""
