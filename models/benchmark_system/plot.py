import csv
import os

import matplotlib
import matplotlib.pyplot as plt
from textwrap import wrap

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def plot_handler(n):
    plot_dictionary = {
        "emoji_frequency_ironic_vs_non_ironic_(ratio)": ["Ironic vs. Non-ironic (ratio)"],
        "emoji_frequency_ironic_vs_non_ironic": ["Ironic vs. Non-ironic"],
        "emoji_frequency_ironic": ["Ironic"],
        "emoji_frequency_non_ironic": ["Non-ironic"],
        "emoji_frequency": ["All"],
        "gram_frequency_ironic_vs_non_ironic_(ratio)": ["Ironic vs. Non-ironic (ratio)"],
        "gram_frequency_ironic_vs_non_ironic": ["Ironic vs. Non-ironic"],
        "gram_frequency_ironic": ["Ironic"],
        "gram_frequency_non_ironic": ["Non-ironic"],
        "gram_frequency": ["All"],
    }

    output_file = f"{DIR_PATH}/output/output_{n}.csv"
    baseline_results = baseline(output_file)

    with open(output_file) as f:
        reader = csv.reader(f)
        result_list = list(reader)

    already_used = []
    for plot_type, plot_description in plot_dictionary.items():
        relevant_results = []
        for result in result_list:
            if plot_type not in result[0] or result in already_used:
                continue

            already_used.append(result)
            relevant_results.append(result)
        plot_dictionary[plot_type].append(relevant_results)
        plot_one(n, relevant_results, [plot_type, plot_description[0]],
                 baseline_results)

    plot_all(n, plot_dictionary, baseline_results)


def plot_one(n, results, plot_list, baseline_results):
    plot_filename, plot_description = plot_list

    sorted_results = sorted(results, key=lambda x: int(x[0].split("_")[-1][:-4]))
    x = [int(result[0].split("_")[-1][:-4]) for result in sorted_results]
    y = [float(result[-1]) for result in sorted_results]
    fig, ax = plt.subplots()
    ax.plot(x, y)

    if "emoji" in plot_description.lower():
        xlabel = f"Number of {n}-emojis removed from tokenised training set"
        title = (f"$F_1$ scores after removing {n}-emojis "
                 f"({plot_description.lower()}) from the default (tokenised) "
                 "training set")
    else:
        xlabel = f"Number of {n}-grams removed from tokenised training set"
        title = (f"$F_1$ scores after removing {n}-grams "
                 f"({plot_description.lower()}) from the default (tokenised) "
                 "training set")

    ax.set(xlabel=xlabel, ylabel="$F_1$ score", title="\n".join(wrap(title, 60)))
    ax.grid()
    matplotlib.pyplot.xticks(x)
    ax.plot(x, [baseline_results[0]] * len(x), label="Default")
    ax.plot(x, [baseline_results[1]] * len(x), label="Default (tokenised)")
    plt.legend(loc="best")
    fig.savefig(f"{DIR_PATH}/figures/{n}-grams/{n}-{plot_filename}.png")


def plot_all(n, plot_dictionary, baseline_results):
    emoji_plots = [value for key, value in plot_dictionary.items()
                   if "emoji" in key]
    ngram_plots = [value for key, value in plot_dictionary.items()
                   if "gram" in key]

    for i, plots in enumerate([emoji_plots, ngram_plots]):
        fig, ax = plt.subplots()
        ax.grid()

        if not i:
            xlabel = f"Number of {n}-emojis removed from tokenised training set"
            title = (f"$F_1$ scores after removing {n}-emojis from the default"
                     " (tokenised) training set")
            plot_filename = f"{n}-emoji_all"
        else:
            xlabel = f"Number of {n}-grams removed from tokenised training set"
            title = (f"$F_1$ scores after removing {n}-grams from the default"
                     " (tokenised) training set")
            plot_filename = f"{n}-gram_all"

        for plot in plots:
            results = plot[1]
            sorted_results = sorted(results, key=lambda x: int(x[0].split("_")[-1][:-4]))

            x = [int(result[0].split("_")[-1][:-4]) for result in sorted_results]
            y = [float(result[-1]) for result in sorted_results]

            ax.plot(x, y, label=plot[0])

        matplotlib.pyplot.xticks(x)
        ax.plot(x, [baseline_results[0]] * len(x), label="Default")
        ax.plot(x, [baseline_results[1]] * len(x), label="Default (tokenised)")
        plt.legend(loc="best")
        ax.set(xlabel=xlabel, ylabel="$F_1$ score", title="\n".join(wrap(title, 60)))
        fig.savefig(f"{DIR_PATH}/figures/{n}-grams/{plot_filename}.png")


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
    for n in range(1, 5):
        plot_handler(n)
