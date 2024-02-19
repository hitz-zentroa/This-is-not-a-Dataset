import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse


def get_result_table(result_dir):
    result_path = Path(result_dir)
    # Collect all test_results.json files
    result_files = list(result_path.rglob("test_results.json"))

    results = []
    result_names = []
    for result_file in result_files:
        with open(result_file, "r", encoding="utf8") as f:
            values = json.load(f)
        accuracy = float(values["all"]["accuracy"])
        results.append(accuracy)
        result_names.append(result_file.parent.name)

    # Prepare data for plotting
    data = sorted(zip(result_names, results), key=lambda x: x[1])
    result_names, results = zip(*data)

    # Define pastel colors based on accuracy values
    colors = [
        "#ff9999" if x < 55 else "#ffff99" if x < 65 else "#99ff99" for x in results
    ]

    # Improve aesthetics with seaborn
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x=results, y=result_names, palette=colors)
    plt.axvline(x=50, color="grey", linestyle="--", linewidth=1)

    # Adding text for "Random baseline"
    plt.text(
        50.0,
        -0.5,
        "Random baseline",
        horizontalalignment="right",
        color="grey",
        fontsize=12,
    )

    plt.xlabel("Accuracy (%)", fontsize=14, labelpad=15)
    plt.ylabel("Model", fontsize=14, labelpad=15)
    plt.title(
        f"Accuracy in 'This Is Not a Dataset' ({result_path.name})", fontsize=16, pad=20
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Customizing the bar labels for better readability, labels always black
    for bar in bars.patches:
        bars.annotate(
            f"{bar.get_width():.1f}%",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color="black",
            fontsize=12,
        )

    plt.tight_layout()  # Adjust the layout to make room for the labels
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get results from a directory")
    parser.add_argument("result_dir", type=str, help="Directory with the results")
    args = parser.parse_args()
    get_result_table(args.result_dir)

# python3 get_results.py "/run/user/1000/gvfs/sftp:host=tximista.ixa.eus,user=igarcia945/tartalo01/users/igarcia945/ikerlariak/This-is-not-a-Dataset/results/fewshot"
# python3 get_results.py /run/user/1000/gvfs/sftp:host=tximista.ixa.eus,user=igarcia945/tartalo01/users/igarcia945/ikerlariak/This-is-not-a-Dataset/results/zero-shot
