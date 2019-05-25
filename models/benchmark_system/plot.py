import csv
import logging
import os
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(level=logging.INFO)


def plot_handler(n, control=False):
    """Function for handling the plot creation of both the single plots, and
    the plots containing multiple graphs.

    :param n: The n in n-gram or n-emoji.
    :param control: Whether the control group should be plotted or not,
                    defaults to False
    """
    # Dictionary used to group the results from different training sets
    plot_dictionary = {
        "emoji_frequency_ironic_vs_non_ironic_(ratio)": ["Ironic vs. Non-ironic (ratio)"],
        "emoji_frequency_ironic_vs_non_ironic": ["Ironic vs. Non-ironic"],
<<<<<<< HEAD
        "emoji_frequency_non_ironic_vs_ironic_(ratio)": ["Non-ironic vs. Ironic"],
        "emoji_frequency_non_ironic_vs_ironic": ["Non-ironic vs. Ironic"],
=======
>>>>>>> 2815e7afa34cb5f723cd75248646a4c4f0fc65c2
        "emoji_frequency_ironic": ["Ironic"],
        "emoji_frequency_non_ironic": ["Non-ironic"],
        "emoji_frequency": ["All"],
        "gram_frequency_ironic_vs_non_ironic_(ratio)": ["Ironic vs. Non-ironic (ratio)"],
        "gram_frequency_ironic_vs_non_ironic": ["Ironic vs. Non-ironic"],
<<<<<<< HEAD
        "gram_frequency_non_ironic_vs_ironic_(ratio)": ["Non-ironic vs. Ironic (ratio)"],
        "gram_frequency_non_ironic_vs_ironic": ["Non-ironic vs. Ironic"],
=======
>>>>>>> 2815e7afa34cb5f723cd75248646a4c4f0fc65c2
        "gram_frequency_ironic": ["Ironic"],
        "gram_frequency_non_ironic": ["Non-ironic"],
        "gram_frequency": ["All"],
    }

    output_file = f"{DIR_PATH}/output/output_{n}.csv"
    if control:
        output_file = f"{DIR_PATH}/control_output/output_{n}.csv"

    baseline_results = baseline()

    with open(output_file) as f:
        reader = csv.reader(f)
        result_list = list(reader)

    already_used = []
    for plot_type, plot_description in plot_dictionary.items():
        relevant_results = []

        # Get all the relevant results for the plot type
        for result in result_list:
            if plot_type not in result[0] or result in already_used:
                continue

            already_used.append(result)
            relevant_results.append(result)

        plot_dictionary[plot_type].append(relevant_results)

        # Plot the single plot, e.g. from 1-gram_frequency_ironic
        plot_one(n, relevant_results, [plot_type, plot_description[0]],
                 baseline_results, control)

    # Plot all the plots from the same n-gram or n-emoji in the same figure
    plot_all(n, plot_dictionary, baseline_results, control)


def plot_one(n, results, plot_list, baseline_results, control=False):
    """Creates a figure with one plot.

    :param n: The n in n-gram or n-emoji.
    :param results: All the results taken from the output file.
    :param plot_list: List containing the plot type and its description.
    :param baseline_results: Results of the default (tokenised) training set.
    :param control: Whether the control group should be plotted or not,
                    defaults to False
    """
    plot_filename, plot_description = plot_list
    logging.info(f"Plotting {n}-{plot_filename}")

    # Sort the results and create x (number of n-grams removed) and y (F1 score)
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

    # Set the labels, grid and plot the baseline results
    ax.set(xlabel=xlabel, ylabel="$F_1$ score", title="\n".join(wrap(title, 60)))
    ax.plot(x, [baseline_results[0]] * len(x), label="Default")
    ax.plot(x, [baseline_results[1]] * len(x), label="Default (tokenised)")
    ax.grid()

    matplotlib.pyplot.xticks(x)
    plt.legend(loc="best")

    figures_directory = "figures"
    if control:
        figures_directory = "control_figures"

    fig.savefig(f"{DIR_PATH}/{figures_directory}/{n}-grams/{n}-{plot_filename}.png")


def plot_all(n, plot_dictionary, baseline_results, control=False):
    """Plots all the results of e.g. 1-grams in one figure.

    :param n: The n in n-gram or n-emoji.
    :param plot_dictionary: Dictionary containing the plot type as key, and
                            a list with the plots description and results as
                            value.
    :param baseline_results: Results of the default (tokenised) training set.
    :param control: Whether the control group should be plotted or not,
                    defaults to False
    """
    # Separate the n-emoji and n-gram plots
    emoji_plots = [value for key, value in plot_dictionary.items()
                   if "emoji" in key]
    ngram_plots = [value for key, value in plot_dictionary.items()
                   if "gram" in key]

    figures_directory = "figures"
    if control:
        figures_directory = "control_figures"

    for i, plots in enumerate([emoji_plots, ngram_plots]):
        fig, ax = plt.subplots()

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

        logging.info(f"Plotting {plot_filename}")

        for plot in plots:
            results = plot[1]
            # Sort the results and create x (number of n-grams removed) and
            # y (F1 score)
            sorted_results = sorted(results, key=lambda x: int(x[0].split("_")[-1][:-4]))
            x = [int(result[0].split("_")[-1][:-4]) for result in sorted_results]
            y = [float(result[-1]) for result in sorted_results]

            ax.plot(x, y, label=plot[0])

        # Set the labels, grid and plot the baseline results
        ax.plot(x, [baseline_results[0]] * len(x), label="Default")
        ax.plot(x, [baseline_results[1]] * len(x), label="Default (tokenised)")
        ax.set(xlabel=xlabel, ylabel="$F_1$ score", title="\n".join(wrap(title, 60)))
        ax.grid()

        plt.legend(loc="best")
        matplotlib.pyplot.xticks(x)

        fig.savefig(f"{DIR_PATH}/{figures_directory}/{n}-grams/{plot_filename}.png")


def baseline():
    """Returns the results of both the default training set and the tokenised
    default training set.
    """
    logging.info("Getting baseline results")

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
    # For normal plots do 1, 2, 3 and 4-grams / emojis
    for n in range(1, 5):
        plot_handler(n)

    # For the control plots, only plot 1-grams
    plot_handler(1, control=True)
