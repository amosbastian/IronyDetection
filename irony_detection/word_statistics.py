import logging
import os
import re

from utils import (count_ngrams, irony_comparison_handler,
                   ngram_removal_handler, parse_dataset, tokenise_default)

logging.basicConfig(level=logging.DEBUG)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def ngram_frequency(corpus, type=None):
    """Creates an n-gram frequency file, with an n-gram and its frequency on
    each line, from the given corpus.

    :param corpus: A list of tweets.
    :param type: Type of corpus provided (e.g. ironic or non-ironic).
    """
    ngrams = count_ngrams(corpus)
    for n, frequencies in ngrams.items():
        filename = f"{n}-gram_frequency"
        if type:
            filename += f"_{type}"

        logging.info(f"Creating n-gram frequency file: {filename}.txt")
        with open(f"{DIR_PATH}/../output/{filename}.txt", "w") as f:
            f.write("Position\tFrequency\tn-gram\n")
            for i, counter in enumerate(frequencies.most_common()):
                ngram, frequency = counter
                f.write(f"{i + 1}\t{frequency}\t{' '.join(ngram)}\n")


def ngram_frequency_handler(labels, corpus):
    """Creates three separate n-gram frequency files:
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

    ngram_frequency(corpus)
    ngram_frequency(ironic, "ironic")
    ngram_frequency(non_ironic, "non_ironic")


def relative_ngram_frequency(filename, ngram_frequencies):
    """Calculates the observed relative frequency, which is typically
    normalised and reported as a frequency per 1,000 or 1,000,000 words, of
    each word in the corpus.

    :param filename: Name of the n-gram frequency file.
    :param ngram_frequencies: List of n-gram, frequency tuples.
    """
    total_ngrams = sum([int(x[1]) for x in ngram_frequencies])
    relative_frequencies = []

    for ngram, frequency in ngram_frequencies:
        # Calculate relative frequency per 1,000 n-grams
        relative_frequency = int(frequency) * 1000.0 / total_ngrams
        relative_frequencies.append((ngram, relative_frequency))

    logging.info(f"Creating relative n-gram frequency file: relative_{filename}")
    with open(f"{DIR_PATH}/../output/relative_{filename}", "w") as f:
        # Sort n-grams by relative frequency (descending)
        f.write("Position\tRelative Frequency\tn-gram\n")
        for i, counter in enumerate(sorted(relative_frequencies,
                                           key=lambda x: x[1],
                                           reverse=True)):
            ngram, frequency = counter
            f.write(f"{i + 1}\t{frequency}\t{ngram}\n")


def relative_ngram_frequency_handler():
    """Creates a relative word frequency file from each normal word frequency
    file in the output directory.
    """
    for filename in os.listdir(f"{DIR_PATH}/../output/"):
        if not re.match(r"\d-gram", filename):
            continue

        with open(os.path.join(f"{DIR_PATH}/../output/", filename)) as f:
            ngram_frequencies = []

            for line in f.read().splitlines():
                if line.lower().startswith("position\t"):
                    continue

                frequency, ngram = line.split("\t")[1:]
                ngram_frequencies.append((ngram, frequency))

            relative_ngram_frequency(filename, ngram_frequencies)


def word_statistics(labels, corpus):
    ngram_frequency_handler(labels, corpus)
    relative_ngram_frequency_handler()


if __name__ == "__main__":
    labels, corpus = parse_dataset("SemEval2018-T3-train-taskA_emoji")
    word_statistics(labels, corpus)
    tokenise_default()
    irony_comparison_handler()
    ngram_removal_handler("SemEval2018-T3-train-taskA_emoji", range(1, 31))
