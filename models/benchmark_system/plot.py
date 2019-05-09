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
    output_file = f"{DIR_PATH}/output/output_1.csv"
    baseline_results = baseline(output_file)

    with open(output_file) as f:
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

        plot(relevant_results, [word_type, description], baseline_results)


def plot(results, plot_list, baseline_results):
    plot_filename, plot_description = plot_list

    sorted_results = sorted(results, key=lambda x: int(x[0].split("_")[-1][:-4]))
    x = [int(result[0].split("_")[-1][:-4]) for result in sorted_results]
    y = [float(result[-1]) for result in sorted_results]
    fig, ax = plt.subplots()
    ax.plot(x, y)

    if "emoji" in plot_description.lower():
        xlabel = "$N$ emojis removed"
    else:
        xlabel = "$N$ words removed"

    ax.set(xlabel=xlabel, ylabel="$F_1$ score",
           title=plot_description)
    ax.grid()
    matplotlib.pyplot.xticks(x)
    ax.plot(x, [baseline_results[0]] * len(x), label="Default")
    ax.plot(x, [baseline_results[1]] * len(x), label="Default (tokenised)")
    plt.legend(loc="best")
    fig.savefig(f"{DIR_PATH}/figures/{plot_filename}.png")
    plt.show()


def baseline(filename):
    tokenised = "SemEval2018-T3-train-taskA_emoji_tokenised.txt"
    default = "SemEval2018-T3-train-taskA_emoji.txt"

    baseline_results = [0, 0]

    with open(f"{DIR_PATH}/output/output_1.csv") as f:
        reader = csv.reader(f)
        for result in list(reader):
            if result[0] == default:
                baseline_results[0] = float(result[-1])
            elif result[0] == tokenised:
                baseline_results[1] = float(result[-1])

    return baseline_results

if __name__ == "__main__":
    plot_handler()
