import os
import csv


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def plot_handler():
    plots = {
        "emoji_frequency_ironic": "",
        "emoji_frequency_non_ironic": "",
        "emoji_frequency_ironic_vs_non_ironic": "",
        "emoji_frequency": "",
        "gram_frequency_ironic": "",
        "gram_frequency_non_ironic": "",
        "gram_frequency_ironic_vs_non_ironic": "",
        "gram_frequency": "",
    }

    with open(f"{DIR_PATH}/output/output_1.csv") as f:
        reader = csv.reader(f)
        result_list = list(reader)

    already_used = []
    for word_type, description in plots.items():
        relevant_results = []
        for result in result_list:
            if word_type not in result[0] or result in already_used:
                continue

            already_used.append(result)
            relevant_results.append(result)

        plot(relevant_results, description)


def plot(results, description):
    for result in sorted(results, key=lambda x: float(x[-1]), reverse=True):
        print(result)


if __name__ == "__main__":
    plot_handler()
