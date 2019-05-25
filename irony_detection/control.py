import logging
import os
from random import shuffle

import nltk

from utils import remove_ngram, tokenise

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def random_ngrams(ngrams, number_of_ngrams):
    relevant_ngrams = ngrams[:number_of_ngrams]
    remaining_ngrams = ngrams[number_of_ngrams:]

    relevant_tagged = nltk.pos_tag(relevant_ngrams)
    remaining_tagged = nltk.pos_tag([i for i in remaining_ngrams if i])

    shuffle(remaining_tagged)

    random_ngrams_list = []
    for _, tag in relevant_tagged:
        random_pos_tag = next(pos_tag for pos_tag in remaining_tagged
                              if pos_tag[1] == tag)
        remaining_tagged.remove(random_pos_tag)
        random_ngrams_list.append(random_pos_tag[0])

    return random_ngrams_list


def ngram_removal(filename, ngrams, n):
    training_directory = f"{DIR_PATH}/../datasets/train/"
    output_directory = f"{DIR_PATH}/../output/"
    dataset_filename = "SemEval2018-T3-train-taskA_emoji.txt"

    logging.info(f"Removing top {n} n-grams from {dataset_filename} using "
                 f"{filename} (CONTROL)")

    out_filename = f"CONTROL-{dataset_filename[:-4]}_{filename[:-4]}_{n}.txt"

    fout = open(f"{training_directory}{out_filename}", "w+")
    fout.write("Tweet index	Label	Tweet text\n")

    with open(f"{training_directory}{dataset_filename}") as f:
        for line in f.readlines():
            if line.lower().startswith("tweet index"):
                continue

            # Tokenise the tweet in the same way as when calculating n-gram
            # frequencies, and replace certain n-grams in it
            tweet = " ".join(tokenise(line.split("\t")[2]))
            for ngram in ngrams:
                tweet = remove_ngram(tweet, ngram, len(ngram.split()))

            # Tweet has been completely removed, so don't include it
            if not tweet:
                continue

            # Write tokenised tweet with n-grams replaced back to a file
            split_line = line.split("\t")
            split_line[2] = tweet
            fout.write("\t".join(split_line) + "\n")


def create_control(filename, ngrams, number_of_ngrams):
    ngrams = random_ngrams(ngrams, number_of_ngrams)

    with open(f"{DIR_PATH}/control.csv", "a") as f:
        f.write(f"{filename},{','.join(ngrams)}\n")

    for n in range(number_of_ngrams):
        ngram_removal(filename, ngrams[:n + 1], n + 1)


def control_handler(n, number_of_ngrams):
    output_directory = f"{DIR_PATH}/../output/"

    for filename in os.listdir(output_directory):
        if not filename.startswith(f"{n}-gram"):
            continue

        ngrams = []
        with open(os.path.join(output_directory, filename)) as f:
            for line in f.read().splitlines():
                if line.lower().startswith("position\t"):
                    continue

                _, ngram = line.split("\t")[1:]
                ngrams.append(ngram)

        create_control(filename, ngrams, number_of_ngrams)


if __name__ == "__main__":
    control_handler(1, 20)
