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
        "all": []
    }

    for filename in os.listdir(training_directory):
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


def frequency_handler():
    default = ["SemEval2018-T3-train-taskA_emoji_tokenised.txt",
               "SemEval2018-T3-train-taskA_emoji.txt",
               "README.md"]

    total_words = count_words(default[0])
    groups = group_training_sets()

    for key, filenames in groups.items():
        fout = open(f"{DIR_PATH}/output/words_removed_{key}.csv", "w+")
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
