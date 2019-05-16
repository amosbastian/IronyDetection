import csv
import logging
import os
import re
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
training_directory = f"{DIR_PATH}/../datasets/train/"


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


def frequency_handler():
    default_training = ["SemEval2018-T3-train-taskA_emoji_tokenised.txt",
                        "SemEval2018-T3-train-taskA_emoji.txt",
                        "README.md"]

    total_words = count_words(default_training[0])

    fout = open(f"{DIR_PATH}/words_removed.csv", "w+")
    fout.write("Training Set,Number of Words Removed,Percentage of Words Removed\n")

    for filename in os.listdir(training_directory):
        if filename in default_training:
            continue

        word_count = count_words(filename)
        percentage_removed = 100 - percentage(word_count, total_words)
        words_removed = total_words - word_count
        fout.write(f"{filename},{words_removed},{percentage_removed}\n")

if __name__ == "__main__":
    frequency_handler()
