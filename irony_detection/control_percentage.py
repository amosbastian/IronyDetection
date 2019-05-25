import logging
import os
from random import shuffle

from utils import remove_ngram, tokenise

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
training_directory = f"{DIR_PATH}/../datasets/train/"
output_directory = f"{DIR_PATH}/../output/"
word_removal_directory = f"{DIR_PATH}/frequency_statistics/output/"

filename_all = "1-gram_frequency.txt"
filename_ironic = "1-gram_frequency_ironic.txt"
filename_non_ironic = "1-gram_frequency_non_ironic.txt"


def get_ngrams(filename):
    ngram_list = []
    with open(f"{output_directory}{filename}") as frequency_file:
        for line in frequency_file.readlines():
            if line.lower().startswith("position\t"):
                continue

            _, ngram = line.split("\t")[1:]
            ngram_list.append(ngram.replace("\n", ""))

    return ngram_list


def ngram_frequencies(filename):
    frequency_list = []
    with open(f"{output_directory}{filename}") as frequency_file:
        for line in frequency_file.readlines():
            if line.lower().startswith("position\t"):
                continue

            frequency, ngram = line.split("\t")[1:]
            frequency_list.append((ngram.replace("\n", ""), int(frequency)))

    return frequency_list


def percentage(percent, whole):
    return round((percent * whole) / 100.0)


def percentages(filename):
    filename = f"SemEval2018-T3-train-taskA_emoji_1-gram_frequency_{filename}"

    lines = []
    with open(f"{word_removal_directory}words_removed_1.csv") as f:
        for line in f.readlines():
            if line.lower().startswith("training set\t"):
                continue

            lines.append(line)

    percentages_list = []
    for i in range(1, 21):
        for line in lines:
            fname = line.split(",")[0]

            if fname != f"{filename}_{i}.txt":
                continue

            percentages_list.append(float(line.split(",")[-1]))

    return percentages_list


def random_words(ngrams, removal_percentages):
    all_frequencies = ngram_frequencies(filename_all)
    frequencies = [x for x in all_frequencies if x[0] in ngrams]

    total_words = sum([x[1] for x in ngram_frequencies(filename_all)])
    word_dictionary = dict((i, []) for i in range(1, 21))

    for i, removal_percentage in enumerate(removal_percentages):
        if i > 0:
            removal_percentage = removal_percentages[i] - removal_percentages[i - 1]
        number_to_remove = percentage(removal_percentage, total_words)
        shuffle(frequencies)

        words_to_remove = []
        for frequency_tuple in frequencies:
            ngram, frequency = frequency_tuple
            if number_to_remove - frequency < 0:
                continue

            words_to_remove.append(ngram)
            number_to_remove -= frequency
            frequencies.remove(frequency_tuple)

        word_dictionary[i + 1] = list(set(words_to_remove))

    return word_dictionary


def ngram_removal(type, ngrams, n):
    dataset_filename = "SemEval2018-T3-train-taskA_emoji.txt"
    out_filename = f"CONTROL-PERCENTAGE-{dataset_filename[:-4]}_{type}_{n}.txt"

    if n > 1:
        dataset_filename = f"CONTROL-PERCENTAGE-{dataset_filename[:-4]}_{type}_{n - 1}.txt"

    logging.info(f"Removing n-grams from {dataset_filename} "
                 "(CONTROL-PERCENTAGE)")

    fout = open(f"{training_directory}{out_filename}", "w+")
    fout.write("Tweet index\tLabel\tTweet text\n")

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

    fout.close()


def save_control_csv(word_dictionary, type):
    with open(f"{DIR_PATH}/control_percentage_{type}.csv", "a") as f:
        for key, value in word_dictionary.items():
            f.write(f"{key},{','.join(value)}\n")


def control_handler():
    ironic_ngrams = get_ngrams(filename_ironic)
    ironic_percentages = percentages("ironic")

    non_ironic_ngrams = get_ngrams(filename_non_ironic)
    non_ironic_percentages = percentages("non_ironic")

    ironic_words = random_words(ironic_ngrams, ironic_percentages)
    non_ironic_words = random_words(non_ironic_ngrams, non_ironic_percentages)

    for n in range(1, 21):
        ngram_removal("ironic", ironic_words[n], n)
        ngram_removal("non_ironic", non_ironic_words[n], n)

    save_control_csv(ironic_words, "ironic")
    save_control_csv(non_ironic_words, "non_ironic")

if __name__ == "__main__":
    control_handler()
