import csv
import logging
import os
import re
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
training_directory = f"{DIR_PATH}/../../datasets/train/"


def percentage(part, whole):
    return 100 * float(part) / float(whole)


def count_words(filename):
    all_words = []

    with open(f"{training_directory}{filename}") as dataset:
        for line in dataset.readlines():
            if line.lower().startswith("tweet index"):
                continue

            line = line.rstrip()
            try:
                tweet = line.split("\t")[2]
            except:
                continue
            all_words.extend(tweet.split())

    return len(all_words)


def group_training_sets():
    groups = {
        "emoji": [],
        "n-grams": [],
        "all": [],
        "control": []
    }

    for filename in os.listdir(training_directory):
        if "CONTROL" in filename:
            groups["control"].append(filename)
            continue

        split_filename = filename.split("_")
        groups["all"].append(filename)

        try:
            element_type = split_filename[2]
            corpus_type = "_".join(split_filename[4:-1])
        except:
            continue

        if "emoji" in element_type:
            groups["emoji"].append(filename)
        else:
            groups["n-grams"].append(filename)

        number = element_type.split("-")[0]
        if number in ["1", "2", "3", "4"]:
            groups.setdefault(number, [])
            groups[number].append(filename)

        if corpus_type:
            groups.setdefault(corpus_type, [])
            groups[corpus_type].append(filename)

    return groups


def plot_handler(n, control=False):
    plot_dictionary = {
        "emoji_frequency_ironic_vs_non_ironic_(ratio)": ["Ironic vs. Non-ironic (ratio)"],
        "emoji_frequency_ironic_vs_non_ironic": ["Ironic vs. Non-ironic"],
        "emoji_frequency_non_ironic_vs_ironic_(ratio)": ["Non-ironic vs. Ironic"],
        "emoji_frequency_non_ironic_vs_ironic": ["Non-ironic vs. Ironic"],
        "emoji_frequency_ironic": ["Ironic"],
        "emoji_frequency_non_ironic": ["Non-ironic"],
        "emoji_frequency": ["All"],
        "gram_frequency_ironic_vs_non_ironic_(ratio)": ["Ironic vs. Non-ironic (ratio)"],
        "gram_frequency_ironic_vs_non_ironic": ["Ironic vs. Non-ironic"],
        "gram_frequency_non_ironic_vs_ironic_(ratio)": ["Non-ironic vs. Ironic (ratio)"],
        "gram_frequency_non_ironic_vs_ironic": ["Non-ironic vs. Ironic"],
        "gram_frequency_ironic": ["Ironic"],
        "gram_frequency_non_ironic": ["Non-ironic"],
        "gram_frequency": ["All"],
    }

    output_file = f"{DIR_PATH}/output/words_removed_{n}.csv"
    if control:
        output_file = f"{DIR_PATH}/control_output/words_removed_control.csv"

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
        plot_one(n, relevant_results, [plot_type, plot_description[0]], control)

    plot_all(n, plot_dictionary, control)


def plot_one(n, results, plot_list, control=False):
    plot_filename, plot_description = plot_list

    sorted_results = sorted(results, key=lambda x: int(x[0].split("_")[-1][:-4]))
    x = [int(result[0].split("_")[-1][:-4]) for result in sorted_results]
    y = [float(result[-1]) for result in sorted_results]
    fig, ax = plt.subplots()
    ax.plot(x, y)

    if "emoji" in plot_description.lower():
        xlabel = f"The number of most frequent {n}-emojis removed"
        title = ("The percentage of words removed from the default "
                 "(tokenised) training set when removing the $N$ most "
                 f"frequent ({plot_description.lower()}) {n}-emojis")
    else:
        xlabel = f"The number of most frequent {n}-grams removed"
        title = ("The percentage of words removed from the default "
                 "(tokenised) training set when removing the $N$ most "
                 f"frequent ({plot_description.lower()}) {n}-grams")

    figures_directory = "figures"
    if control:
        figures_directory = "control_figures"
        title = ("The percentage of words removed from the default "
                 "(tokenised) training set when removing $N$ random 1-grams")

    ax.set(xlabel=xlabel, ylabel="% words removed", title="\n".join(wrap(title, 60)))
    ax.grid()
    matplotlib.pyplot.xticks(x)

    fig.savefig(f"{DIR_PATH}/{figures_directory}/{n}-grams/{n}-{plot_filename}.png")


def plot_all(n, plot_dictionary, control=False):
    emoji_plots = [value for key, value in plot_dictionary.items()
                   if "emoji" in key]
    ngram_plots = [value for key, value in plot_dictionary.items()
                   if "gram" in key]

    figures_directory = "figures"

    for i, plots in enumerate([emoji_plots, ngram_plots]):
        fig, ax = plt.subplots()
        ax.grid()

        if not i:
            xlabel = f"The number of most frequent {n}-emojis removed"
            title = ("The percentage of words removed from the default "
                     "(tokenised) training set when removing the $N$ most "
                     f"frequent {n}-emojis")
            plot_filename = f"{n}-emoji_all"
        else:
            xlabel = f"The number of most frequent {n}-grams removed"
            title = ("The percentage of words removed from the default "
                     "(tokenised) training set when removing the $N$ most "
                     f"frequent {n}-grams")
            plot_filename = f"{n}-gram_all"

        if control:
            figures_directory = "control_figures"
            title = ("The percentage of words removed from the default "
                     "(tokenised) training set when removing $N$ random "
                     "1-grams")

        for plot in plots:
            results = plot[1]
            sorted_results = sorted(results, key=lambda x: int(x[0].split("_")[-1][:-4]))

            x = [int(result[0].split("_")[-1][:-4]) for result in sorted_results]
            y = [float(result[-1]) for result in sorted_results]

            ax.plot(x, y, label=plot[0])

        matplotlib.pyplot.xticks(x)
        plt.legend(loc="best")
        ax.set(xlabel=xlabel, ylabel="% words removed", title="\n".join(wrap(title, 60)))
        fig.savefig(f"{DIR_PATH}/{figures_directory}/{n}-grams/{plot_filename}.png")


def frequency_handler():
    default = ["SemEval2018-T3-train-taskA_emoji_tokenised.txt",
               "SemEval2018-T3-train-taskA_emoji.txt",
               "README.md"]

    total_words = count_words(default[0])
    groups = group_training_sets()

    for key, filenames in groups.items():
        output_directory = "output"
        if "control" in key:
            output_directory = "control_output"

        fout = open(f"{DIR_PATH}/{output_directory}/words_removed_{key}.csv", "w+")
        fout.write("Training Set,Number of Words Removed,Percentage of Words Removed\n")

        for filename in filenames:
            if filename in default:
                continue

            word_count = count_words(filename)
            percentage_removed = 100 - percentage(word_count, total_words)
            words_removed = total_words - word_count
            fout.write(f"{filename},{words_removed},{percentage_removed}\n")

if __name__ == "__main__":
    frequency_handler()
    for n in range(1, 5):
        plot_handler(n)

    plot_handler(1, True)
