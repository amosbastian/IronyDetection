import logging
import os

import emoji
import numpy as np
from emoji.unicode_codes import EMOJI_UNICODE

from utils import count_ngrams, parse_dataset

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

        logging.info(f"Creating n-gram frequency file: {filename}.txt")
        with open(f"{DIR_PATH}/../output/{filename}.txt", "w") as f:
            f.write("Position\tFrequency\tn-gram\n")
            for i, counter in enumerate(frequencies.most_common()):
                ngram, frequency = counter

                if not set(ngram).issubset(set(EMOJI_UNICODE.keys())):
                    continue

                f.write(f"{i + 1}\t{frequency}\t{' '.join(ngram)}\n")


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


if __name__ == "__main__":
    labels, corpus = parse_dataset("SemEval2018-T3-train-taskA_emoji")
    emoji_frequency_handler(labels, corpus)
    # ngram_frequency_handler(labels, corpus)
    # relative_ngram_frequency_handler()
    # ngram_removal_handler("SemEval2018-T3-train-taskA_emoji", range(31))
    # tokenise_default()
