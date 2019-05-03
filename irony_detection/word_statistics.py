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
from nltk.corpus import stopwords
from nltk.probability import FreqDist

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

logging.basicConfig(level=logging.DEBUG)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


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
            if word not in stopwords.words("english")]


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
                tweet = tweet.replace(ngram, "").strip()

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
    with open(os.path.join(f"{DIR_PATH}/../output/", filename)) as f:
        ngram_frequencies = []

        for line in f.read().splitlines():
            if line.lower().startswith("position\t"):
                continue

            frequency, ngram = line.split("\t")[1:]
            ngram_frequencies.append((ngram, frequency))

    return ngram_frequencies


def frequency_difference(irony, non_irony):
    frequency_difference = []
    for ngram, frequency in non_irony:
        try:
            ironic_frequency = int([x for x in irony if x[0] == ngram][-1][-1])
        except IndexError:
            ironic_frequency = 0

        frequency_difference.append(
            (ngram, abs(int(frequency) - ironic_frequency)))

    return sorted(frequency_difference, key=lambda x: x[1], reverse=True)


def irony_comparison():
    for n in range(1, 5):
        irony = get_frequencies(f"{n}-gram_frequency_ironic.txt")
        non_irony = get_frequencies(f"{n}-gram_frequency_non_ironic.txt")
        filename = f"{n}-gram_frequency_ironic_vs_non_ironic"

        logging.info(f"Creating n-gram frequency file: {filename}.txt")
        with open(f"{DIR_PATH}/../output/{filename}.txt", "w") as f:
            f.write("Position\tFrequency\tn-gram\n")
            for i, counter in enumerate(frequency_difference(irony, non_irony)):
                ngram, frequency = counter
                f.write(f"{i + 1}\t{frequency}\t{ngram}\n")


if __name__ == "__main__":
    # labels, corpus = parse_dataset("SemEval2018-T3-train-taskA_emoji")
    # ngram_frequency_handler(labels, corpus)
    # relative_ngram_frequency_handler()
    # ngram_removal_handler("SemEval2018-T3-train-taskA_emoji", [3, 5, 10])
    # tokenise_default()
    irony_comparison()
