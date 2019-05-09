import csv
import os

import matplotlib
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def plot_handler():
    plots = {
        "emoji_frequency_ironic_vs_non_ironic": "1-emoji - Ironic vs. Non-ironic",
        "emoji_frequency_ironic": "1-emoji - Ironic",
        "emoji_frequency_non_ironic": "1-emoji - Non-ironic",
        "emoji_frequency": "1-emoji - All",
        "gram_frequency_ironic_vs_non_ironic": "1-gram - Ironic vs. Non-ironic",
        "gram_frequency_ironic": "1-gram - Ironic",
        "gram_frequency_non_ironic": "1-gram - Non-ironic",
        "gram_frequency": "1-gram - All",
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
    sorted_results = sorted(results, key=lambda x: int(x[0].split("_")[-1][:-4]))
    x = [int(result[0].split("_")[-1][:-4]) for result in sorted_results]
    y = [float(result[-1]) for result in sorted_results]
    fig, ax = plt.subplots()
    ax.plot(x, y)

    if "emoji" in description.lower():
        xlabel = "$N$ emojis removed"
    else:
        xlabel = "$N$ words removed"

    ax.set(xlabel=xlabel, ylabel="$F_1$ score",
           title=description)
    ax.grid()
    matplotlib.pyplot.xticks(x)
    plt.show()


if __name__ == "__main__":
    plot_handler()
