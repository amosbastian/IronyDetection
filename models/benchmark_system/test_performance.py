
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


def featurize(corpus):
    """
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    """

    tokenizer = TweetTokenizer(
        preserve_case=False,
        reduce_len=True,
        strip_handles=True).tokenize

    vectorizer = TfidfVectorizer(strip_accents="unicode",
                                 analyzer="word",
                                 tokenizer=tokenizer,
                                 stop_words="english")

    X = vectorizer.fit_transform(corpus)

    return X


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
    training_directory = f"{DIR_PATH}/../../datasets/train/"
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


def test_control(training_directory):
    for filename in os.listdir(training_directory):
        if "CONTROL" not in filename or "PERCENTAGE" in filename:
            continue

        p_filename = f"{DIR_PATH}/control_predictions/predictions_{filename}"
        PREDICTIONSFILE = open(p_filename, "w")
        logging.info(f"Training with dataset: {filename}")

        corpus, y = parse_dataset(f"{training_directory}{filename}")
        X = featurize(corpus)

        predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)

        for p in predicted:
            PREDICTIONSFILE.write("{}\n".format(p))
        PREDICTIONSFILE.close()


def control_percentage_output():
    predictions_directory = "control_percentage_predictions"
    output_directory = "control_percentage_output"
    fout = open(f"{DIR_PATH}/{output_directory}/output_1.csv", "w+")
    fout.write("Training Set,Accuracy,Precision,Recall,F1\n")

    for filename in os.listdir(f"{DIR_PATH}/{predictions_directory}"):
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

    fout.close()


def test_control_percentage(training_directory):
    for filename in os.listdir(training_directory):
        if "PERCENTAGE" not in filename:
            continue

        p_filename = f"{DIR_PATH}/control_percentage_predictions/predictions_{filename}"
        PREDICTIONSFILE = open(p_filename, "w")
        logging.info(f"Training with dataset: {filename}")

        corpus, y = parse_dataset(f"{training_directory}{filename}")
        X = featurize(corpus)

        predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)

        for p in predicted:
            PREDICTIONSFILE.write("{}\n".format(p))
        PREDICTIONSFILE.close()


def test_performance(training_directory):
    for filename in os.listdir(training_directory):
        if "README" in filename or "CONTROL" in filename:
            continue

        p_filename = f"{DIR_PATH}/predictions/predictions_{filename}"
        PREDICTIONSFILE = open(p_filename, "w")
        logging.info(f"Training with dataset: {filename}")

        # Loading dataset and featurised simple Tfidf-BoW model
        corpus, y = parse_dataset(f"{training_directory}{filename}")
        X = featurize(corpus)

        # Returns an array of the same size as `y` where each entry is a
        # prediction obtained by cross validated
        predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)

        for p in predicted:
            PREDICTIONSFILE.write("{}\n".format(p))
        PREDICTIONSFILE.close()


if __name__ == "__main__":
    # 10-fold crossvalidation
    K_FOLDS = 10
    # The default, non-parameter optimized linear-kernel SVM
    CLF = LinearSVC()
    training_directory = f"{DIR_PATH}/../../datasets/train/"
    # test_performance(training_directory)
    test_control(training_directory)

    # groups = group_predictions()
    # create_output(groups)

    control_groups = group_predictions(True)
    create_output(control_groups, True)
    test_control_percentage(training_directory)
    control_percentage_output()
