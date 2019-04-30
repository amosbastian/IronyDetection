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

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


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


def word_frequency(corpus, filename="word_frequency"):
    """Creates a word frequency file, with a word and its frequency on each
    line, from the given corpus.

    :param corpus: A list of tweets.
    :param filename: Name of the file, defaults to "word_frequency".
    """
    words = []
    for sentence in corpus:
        words.extend(tokenise(sentence))

    with open(f"{DIR_PATH}/../output/{filename}.txt", "w") as f:
        for word, frequency in FreqDist(words).most_common():
            f.write(f"{word}, {frequency}\n")


def word_frequency_handler(labels, corpus):
    """Creates three separate word frequency files:
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

    word_frequency(corpus)
    word_frequency(ironic, "word_frequency_ironic")
    word_frequency(non_ironic, "word_frequency_non_ironic")


def relative_word_frequency(filename, word_frequencies):
    """Calculates the observed relative frequency, which is typically
    normalised and reported as a frequency per 1,000 or 1,000,000 words, of
    each word in the corpus.

    :param filename: Name of the word frequency file.
    :param word_frequencies: List of word, frequency tuples.
    """
    total_words = sum([int(x[1]) for x in word_frequencies])
    relative_frequencies = []

    for word, frequency in word_frequencies:
        # Calculate relative frequency per 1,000 words
        relative_frequency = int(frequency) * 1000.0 / total_words
        relative_frequencies.append((word, relative_frequency))

    with open(f"{DIR_PATH}/../output/relative_{filename}", "w") as f:
        # Sort words by relative frequency (descending)
        for word, frequency in sorted(relative_frequencies,
                                      key=lambda x: x[1],
                                      reverse=True):
            f.write(f"{word}, {frequency}\n")


def relative_word_frequency_handler():
    """Creates a relative word frequency file from each normal word frequency
    file in the output directory.
    """
    for filename in os.listdir(f"{DIR_PATH}/../output/"):
        if not filename.startswith("word_frequency"):
            continue

        with open(os.path.join(f"{DIR_PATH}/../output/", filename)) as f:
            word_frequencies = []

            for line in f.read().splitlines():
                word, frequency = line.split(", ")
                word_frequencies.append((word, frequency))

            relative_word_frequency(filename, word_frequencies)


def word_removal(dataset_filename, frequency_filename, number_of_words):
    """Takes the top `number_of_words` words from the given word frequency
    file, removes these from the given dataset, and creates a new dataset from
    this.

    :param dataset_filename: Name of the dataset file.
    :param frequency_filename: Name of the word frequency file.
    :param number_of_words: The number of words that should be used from the
        word frequency file.
    """
    training_directory = f"{DIR_PATH}/../datasets/train/"
    output_directory = f"{DIR_PATH}/../output/"

    # Get the `number_of_words` most frequently occurring words from the
    # frequency file
    with open(f"{output_directory}{frequency_filename}.txt") as f:
        head = [next(f) for x in range(number_of_words)]
        words = [x.split(",")[0] for x in head]

    out_filename = f"{dataset_filename}_{frequency_filename}_{number_of_words}"

    fout = open(f"{training_directory}{out_filename}.txt", "w+")
    fout.write("Tweet index	Label	Tweet text\n")

    with open(f"{training_directory}{dataset_filename}.txt") as f:
        for line in f.readlines():
            if line.lower().startswith("tweet index"):
                continue

            # Tokenise the tweet in the same way as when calculating word
            # frequencies, and replace certain words in it
            tweet = " ".join(tokenise(line.split("\t")[2]))
            for word in words:
                tweet = tweet.replace(word, "")

            # Write tokenised tweet with words replaced back to a file
            split_line = line.split("\t")
            split_line[2] = tweet
            fout.write("\t".join(split_line) + "\n")

if __name__ == "__main__":
    labels, corpus = parse_dataset("SemEval2018-T3-train-taskA_emoji")

    word_frequency_handler(labels, corpus)
    relative_word_frequency_handler()
    word_removal("SemEval2018-T3-train-taskA", "word_frequency", 20)
