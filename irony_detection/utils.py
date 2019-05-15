import collections
import logging
import os
import re
import string

import emoji
import numpy as np
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist

logging.basicConfig(level=logging.DEBUG)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

text_processor = TextPreProcessor(
    normalize=[],
    annotate={},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=False,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)


def count_ngrams(lines, min_length=1, max_length=4):
    """Iterate through given lines iterator (file object or list of lines) and
    return n-gram frequencies. The return value is a dict mapping the length of
    the n-gram to a collections.Counter object of n-gram tuple and number of
    times that n-gram occurred. Returned dict includes n-grams of length
    min_length to max_length.

    Source: https://gist.github.com/benhoyt/dfafeab26d7c02a52ed17b6229f0cb52

    :param lines: File object or list of lines.
    :param min_length: Minimum length of n-gram, defaults to 2.
    :param max_length: Maximum length of n-gram, defaults to 4.
    :return: [description]
    """
    logging.info("Counting n-grams")

    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenise(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams


def tokenise(sentence):
    """Tokenise the given sentence."""
    # Remove punctuation, except #
    punctuation = "!\"$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    sentence = sentence.translate(str.maketrans("", "", punctuation))

    # Remove numbers
    sentence = re.sub(r"\d+", "", sentence)

    # Remove stopwords and convert emojis to text
    processed_sentence = text_processor.pre_process_doc(sentence)
    return [emoji.demojize(word) for word in processed_sentence
            if word not in stopwords.words("english") and word != "#"]


def parse_dataset(training_set):
    """Parses the SemEval dataset, and separates labels and tweets.

    :param training_set: A SemEval dataset containing labels and tweets.
    :return: A list of labels and a list of tweets.
    """
    logging.info("Parsing the dataset")

    training_directory = f"{DIR_PATH}/../datasets/train/"
    labels = []
    corpus = []

    with open(f"{training_directory}{training_set}.txt") as dataset:
        for line in dataset.readlines():
            if line.lower().startswith("tweet index"):
                continue

            line = line.rstrip()
            label = int(line.split("\t")[1])
            tweet = line.split("\t")[2]

            labels.append(label)
            corpus.append(tweet)

    return labels, corpus


def remove_ngram(sentence, ngram_to_remove, n):
    """Removes all occurrences of an n-gram from a sentence.

    :param sentence: The sentence.
    :param ngram_to_remove: The n-gram that will be removed from the sentence.
    :param n: The n in n-gram.
    """
    # If the number of words in the sentence is < than the size of the n-gram,
    # then simply try replacing the n-gram with nothing
    if len(sentence.split()) <= n:
        return sentence.replace(ngram_to_remove, "")

    # Convert sentence into a list of n-grams
    ngram_list = [ngram for ngram in list(ngrams(sentence.split(), n))
                  if " ".join(ngram) != ngram_to_remove]

    # Convert the list of n-grams back to a sentence by taking the first word
    # of each n-gram, and the entirety of the final n-gram
    first_elements = [ngram[0] for ngram in ngram_list[:-1]]
    last_elements = " ".join(ngram_list[-1])
    first_elements.append(last_elements)

    return " ".join(first_elements)


def ngram_removal(dataset_filename, frequency_filename, n):
    """Takes the top `n` n-grams from the given ngram frequency file, removes
    these from the given dataset, and creates a new dataset from this.

    :param dataset_filename: Name of the dataset file.
    :param frequency_filename: Name of the n-gram frequency file.
    :param n: The number of n-grams that should be used from the
        n-gram frequency file.
    """
    logging.info(f"Removing top {n} n-grams from {dataset_filename} using "
                 f"{frequency_filename}")

    training_directory = f"{DIR_PATH}/../datasets/train/"
    output_directory = f"{DIR_PATH}/../output/"

    # Get the `n` most frequently occurring ngrams from the frequency file
    with open(f"{output_directory}{frequency_filename}") as f:
        head = [next(f) for x in range(n)]
        ngrams = [x.split("\t")[-1].replace("\n", "") for x in head[1:]]

    out_filename = f"{dataset_filename[:-4]}_{frequency_filename[:-4]}_{n}.txt"

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
                tweet = remove_ngram(tweet, ngram, n)

            # Tweet has been completely removed, so don't include it
            if not tweet:
                continue

            # Write tokenised tweet with n-grams replaced back to a file
            split_line = line.split("\t")
            split_line[2] = tweet
            fout.write("\t".join(split_line) + "\n")


def ngram_removal_handler(dataset_filename, number_of_ngrams):
    """Uses each file in the output folder, and the `number_of_ngrams` list to
    create new training sets.

    :param dataset_filename: Name of the dataset of which n-grams will be
        removed.
    :param number_of_ngrams: List of numbers indicating how many of the top `n`
        n-grams should be removed from the dataset.
    """
    for n in number_of_ngrams:
        for filename in os.listdir(f"{DIR_PATH}/../output/"):
            if not filename.startswith("relative"):
                ngram_removal(f"{dataset_filename}.txt", filename, n)


def tokenise_default():
    """Tokenises the default training set and writes it back to a file without
    removing any n-grams.
    """
    training_directory = f"{DIR_PATH}/../datasets/train/"
    default_dataset = "SemEval2018-T3-train-taskA_emoji"

    fout = open(f"{training_directory}{default_dataset}_tokenised.txt", "w+")
    fout.write("Tweet index	Label	Tweet text\n")

    with open(f"{training_directory}{default_dataset}.txt") as f:
        for line in f.readlines():
            if line.lower().startswith("tweet index"):
                continue

            # Tokenise the tweet
            tweet = " ".join(tokenise(line.split("\t")[2]))

            # Write tokenised tweet back to a file
            split_line = line.split("\t")
            split_line[2] = tweet
            fout.write("\t".join(split_line) + "\n")


def get_frequencies(filename):
    """Loads n-grams and frequencies from the given file and returns this as a
    list of n-gram, frequency tuples.

    :param filename: The name of the n-gram frequency file.
    """
    with open(os.path.join(f"{DIR_PATH}/../output/", filename)) as f:
        ngram_frequencies = []

        for line in f.read().splitlines():
            if line.lower().startswith("position\t"):
                continue

            frequency, ngram = line.split("\t")[1:]
            ngram_frequencies.append((ngram, frequency))

    return ngram_frequencies


def relative_frequency_ratio(irony, non_irony):
    """Returns a list of tuples containing an n-gram and its relative frequency
    ratio, which is the quotient of the relative frequencies of the n-gram
    in the ironic and non-ironic corpora respectively.

    :param irony: A list of n-gram, frequency tuples generated from ironic
        tweets.
    :param non_irony: A list of n-gram, frequency tuples generated from
        non-ironic tweets.
    """
    frequency_ratios = []

    for ngram, non_ironic_frequency in non_irony:
        # For each n-gram in non-ironic tweets, find its respective frequency
        # in ironic tweets
        try:
            ironic_frequency = float([x for x in irony
                                      if x[0] == ngram][-1][-1])
        except IndexError:
            ironic_frequency = 0

        # Compute the frequency ratio
        frequency_ratio = ironic_frequency / float(non_ironic_frequency)
        frequency_ratios.append((ngram, frequency_ratio))

    # Sort from high to low
    return sorted(frequency_ratios, key=lambda x: x[1], reverse=True)


def frequency_difference(irony, non_irony):
    """Returns a list of tuples containing an n-gram and the difference in
    relative frequency between the ironic and non-ironic tweets.

    :param irony: A list of n-gram, frequency tuples generated from ironic
        tweets.
    :param non_irony: A list of n-gram, frequency tuples generated from
        non-ironic tweets.
    """
    frequency_difference = []

    for ngram, frequency in non_irony:
        # For each n-gram in non-ironic tweets, find its respective frequency
        # in ironic tweets
        try:
            ironic_frequency = float([x for x in irony if x[0] == ngram][-1][-1])
        except IndexError:
            ironic_frequency = 0

        # Take the absolute difference
        frequency_difference.append(
            (ngram, abs(float(frequency) - ironic_frequency)))

    # Sort from high to low
    return sorted(frequency_difference, key=lambda x: x[1], reverse=True)


def irony_comparison_handler(element):
    """Creates new frequency files for each ironic and non-ironic [1-4]-gram
    frequency file.
    """
    for n in range(1, 5):
        # Get the ironic and non-ironic n-gram frequencies
        irony = get_frequencies(f"relative_{n}-{element}_frequency_ironic.txt")
        non_irony = get_frequencies(f"relative_{n}-{element}_frequency_non_ironic.txt")
        filename = f"{n}-{element}_frequency_ironic_vs_non_ironic"

        logging.info(f"Creating n-{element} frequency file: {filename}.txt")
        with open(f"{DIR_PATH}/../output/{filename}.txt", "w") as f:
            f.write(f"Position\tFrequency\tn-{element}\n")
            # Iterate over the n-gram / emoji, relative frequency differences
            # tuples
            for i, counter in enumerate(
                    frequency_difference(irony, non_irony)):
                element_type, frequency = counter
                f.write(f"{i + 1}\t{frequency}\t{element_type}\n")

        with open(f"{DIR_PATH}/../output/{filename}_ratio.txt", "w") as f:
            f.write(f"Position\tFrequency\tn-{element}\n")
            # Iterate over the n-gram / emoji, frequency ratio tuples
            for i, counter in enumerate(
                    relative_frequency_ratio(irony, non_irony)):
                element_type, frequency_ratio = counter
                f.write(f"{i + 1}\t{frequency_ratio}\t{element_type}\n")
