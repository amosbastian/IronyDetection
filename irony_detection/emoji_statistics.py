import logging
import os
import re

import emoji
import numpy as np
from emoji.unicode_codes import EMOJI_UNICODE

from utils import (count_ngrams, irony_comparison_handler,
                   ngram_removal_handler, parse_dataset, tokenise_default)

logging.basicConfig(level=logging.DEBUG)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def emoji_frequency(corpus, type=None):
    """Creates an emoji frequency file, with an n-emoji and its frequency on
    each line, from the given corpus.

    :param corpus: A list of tweets.
    :param type: Type of corpus provided (e.g. ironic or non-ironic).
    """
    ngrams = count_ngrams(corpus)
    for n, frequencies in ngrams.items():
        filename = f"{n}-emoji_frequency"
        if type:
            filename += f"_{type}"

        logging.info(f"Creating emoji frequency file: {filename}.txt")
        with open(f"{DIR_PATH}/../output/{filename}.txt", "w") as f:
            f.write("Position\tFrequency\temoji\n")
            for i, counter in enumerate(frequencies.most_common()):
                emoji, frequency = counter

                if not set(emoji).issubset(set(EMOJI_UNICODE.keys())):
                    continue

                f.write(f"{i + 1}\t{frequency}\t{' '.join(emoji)}\n")


def emoji_frequency_handler(labels, corpus):
    """Creates three separate emoji frequency files:
        1. For the entire corpus.
        2. For all the ironic tweets.
        3. For all the non-ironic tweets.

    :param labels: List of labels (ironic or non-ironic) for the given tweets.
    :param corpus: List of tokenised tweets.
    """
    ironic = []
    non_ironic = []

    for label, sentence in zip(labels, corpus):
        if label:
            ironic.append(sentence)
        else:
            non_ironic.append(sentence)

    emoji_frequency(corpus)
    emoji_frequency(ironic, "ironic")
    emoji_frequency(non_ironic, "non_ironic")


def relative_emoji_frequency(filename, emoji_frequencies):
    """Calculates the observed relative frequency, which is typically
    normalised and reported as a frequency per 1,000 or 1,000,000 words, of
    each word in the corpus.

    :param filename: Name of the emoji frequency file.
    :param emoji_frequencies: List of emoji, frequency tuples.
    """
    total_emoji = sum([int(x[1]) for x in emoji_frequencies])
    relative_frequencies = []

    for emoji, frequency in emoji_frequencies:
        # Calculate relative frequency per 1,000 emojis
        relative_frequency = int(frequency) * 1000.0 / total_emoji
        relative_frequencies.append((emoji, relative_frequency))

    logging.info(f"Creating relative emoji frequency file: relative_{filename}")
    with open(f"{DIR_PATH}/../output/relative_{filename}", "w") as f:
        # Sort emojis by relative frequency (descending)
        f.write("Position\tRelative Frequency\temoji\n")
        for i, counter in enumerate(sorted(relative_frequencies,
                                           key=lambda x: x[1],
                                           reverse=True)):
            emoji, frequency = counter
            f.write(f"{i + 1}\t{frequency}\t{emoji}\n")


def relative_emoji_frequency_handler():
    """Creates a relative emoji frequency file from each normal emoji frequency
    file in the output directory.
    """
    for filename in os.listdir(f"{DIR_PATH}/../output/"):
        if not re.match(r"\d-emoji", filename):
            continue

        with open(os.path.join(f"{DIR_PATH}/../output/", filename)) as f:
            emoji_frequencies = []

            for line in f.read().splitlines():
                if line.lower().startswith("position\t"):
                    continue

                frequency, emoji = line.split("\t")[1:]
                emoji_frequencies.append((emoji, frequency))

            relative_emoji_frequency(filename, emoji_frequencies)


def emoji_statistics(labels, corpus):
    emoji_frequency_handler(labels, corpus)
    relative_emoji_frequency_handler()


if __name__ == "__main__":
    labels, corpus = parse_dataset("SemEval2018-T3-train-taskA_emoji")
    emoji_statistics(labels, corpus)
    tokenise_default()
    irony_comparison_handler("emoji")
    ngram_removal_handler("SemEval2018-T3-train-taskA_emoji", range(1, 31))
