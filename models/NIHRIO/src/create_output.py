

"""
test_performance.py
Uses the benchmark system for the SemEval-2018 Task 3 on Irony detection in
English tweets. The system makes use of token unigrams as features and outputs
cross-validated F1-score. Performance of the model when trained with each
different training set is logged to a file. Adapted from Gilles Jacobs &
Cynthia Van Hee's example.py.
"""

import codecs
import logging
import os

import numpy as np
from nltk.tokenize import TweetTokenizer

from sklearn import metrics
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def parse_dataset(fp):
    """
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    """
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"):
                line = line.rstrip()
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y


def group_predictions(control=False):
    groups = {
        "emoji": [],
        "n-grams": [],
        "all": []
    }

    directory = "predictions"
    if control:
        del groups["emoji"]
        directory = "control_predictions"

    for filename in os.listdir(f"{DIR_PATH}/{directory}/"):
        split_filename = filename.split("_")
        groups["all"].append(filename)

        try:
            element_type = split_filename[3]
            corpus_type = "_".join(split_filename[5:-1])
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


def create_output(groups, control=False):
    training_directory = f"{DIR_PATH}/../../../datasets/train/"
    default = ["predictions_SemEval2018-T3-train-taskA_emoji.txt",
               "predictions_SemEval2018-T3-train-taskA_emoji_tokenised.txt"]

    output_directory = "output"
    predictions_directory = "predictions"
    if control:
        output_directory = "control_output"
        predictions_directory = "control_predictions"

    for key, filenames in groups.items():
        fout = open(f"{DIR_PATH}/{output_directory}/output_{key}.csv", "w+")
        fout.write("Training Set,Accuracy,Precision,Recall,F1\n")
        filenames.extend(default)

        for filename in set(filenames):
            training_set = filename.replace('predictions_', '')
            _, y = parse_dataset(f"{training_directory}{training_set}")
            with open(f"{DIR_PATH}/{predictions_directory}/{filename}") as f:
                predictions = [int(prediction) for prediction in f]

            # Get performance
            accuracy = metrics.accuracy_score(y, predictions)
            precision = metrics.precision_score(y, predictions, pos_label=1)
            recall = metrics.recall_score(y, predictions, pos_label=1)
            f1_score = metrics.f1_score(y, predictions, pos_label=1)

            fout.write(f"{training_set},{accuracy},{precision},{recall},{f1_score}\n")


if __name__ == "__main__":
    groups = group_predictions()
    create_output(groups)