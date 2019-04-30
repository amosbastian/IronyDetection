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
import collections

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

        with open(f"{DIR_PATH}/../output/{filename}.txt", "w") as f:
            for ngram, frequency in frequencies.most_common():
                f.write(f"{' '.join(ngram)}, {frequency}\n")


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
        # Calculate relative frequency per 1,000 ngrams
        relative_frequency = int(frequency) * 1000.0 / total_ngrams
        relative_frequencies.append((ngram, relative_frequency))

    with open(f"{DIR_PATH}/../output/relative_{filename}", "w") as f:
        # Sort ngrams by relative frequency (descending)
        for ngram, frequency in sorted(relative_frequencies,
                                       key=lambda x: x[1],
                                       reverse=True):
            f.write(f"{ngram}, {frequency}\n")


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
                ngram, frequency = line.split(", ")
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
    training_directory = f"{DIR_PATH}/../datasets/train/"
    output_directory = f"{DIR_PATH}/../output/"

    # Get the `n` most frequently occurring ngrams from the frequency file
    with open(f"{output_directory}{frequency_filename}.txt") as f:
        head = [next(f) for x in range(n)]
        ngrams = [x.split(",")[0] for x in head]

    out_filename = f"{dataset_filename}_{frequency_filename}_{n}"

    fout = open(f"{training_directory}{out_filename}.txt", "w+")
    fout.write("Tweet index	Label	Tweet text\n")

    with open(f"{training_directory}{dataset_filename}.txt") as f:
        for line in f.readlines():
            if line.lower().startswith("tweet index"):
                continue

            # Tokenise the tweet in the same way as when calculating n-gram
            # frequencies, and replace certain n-grams in it
            tweet = " ".join(tokenise(line.split("\t")[2]))
            for ngram in ngrams:
                tweet = tweet.replace(ngram, "")

            # Write tokenised tweet with n-grams replaced back to a file
            split_line = line.split("\t")
            split_line[2] = tweet
            fout.write("\t".join(split_line) + "\n")

if __name__ == "__main__":
    labels, corpus = parse_dataset("SemEval2018-T3-train-taskA_emoji")
    ngram_frequency_handler(labels, corpus)
    relative_ngram_frequency_handler()
    ngram_removal("SemEval2018-T3-train-taskA", "2-gram_frequency", 20)
