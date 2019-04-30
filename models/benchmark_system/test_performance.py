
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


if __name__ == "__main__":
    # 10-fold crossvalidation
    K_FOLDS = 10
    # The default, non-parameter optimized linear-kernel SVM
    CLF = LinearSVC()
    training_directory = f"{DIR_PATH}/../../datasets/train/"

    fout = open(f"{DIR_PATH}/output.csv", "w+")
    fout.write("Training Set,Accuracy,Precision,Recall,F1\n")

    for filename in os.listdir(training_directory):
        if "README" in filename:
            continue

        PREDICTIONSFILE = open(f"{DIR_PATH}/predictions/predictions_{filename}", "w")
        logging.info(f"Training with dataset: {filename}")

        # Loading dataset and featurised simple Tfidf-BoW model
        corpus, y = parse_dataset(f"{training_directory}{filename}")
        X = featurize(corpus)

        class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()

        # Returns an array of the same size as `y` where each entry is a
        # prediction obtained by cross validated
        predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)

        # Get performance
        accuracy = metrics.accuracy_score(y, predicted)
        precision = metrics.precision_score(y, predicted, pos_label=1)
        recall = metrics.recall_score(y, predicted, pos_label=1)
        f1_score = metrics.f1_score(y, predicted, pos_label=1)

        fout.write(f"{filename},{accuracy},{precision},{recall},{f1_score}\n")

        for p in predicted:
            PREDICTIONSFILE.write("{}\n".format(p))
        PREDICTIONSFILE.close()
