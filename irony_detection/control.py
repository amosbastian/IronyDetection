import os
from random import shuffle
import nltk

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def create_control(ngrams, number_of_ngrams):
    relevant_ngrams = ngrams[:number_of_ngrams]
    remaining_ngrams = ngrams[number_of_ngrams:]

    relevant_tagged = nltk.pos_tag(relevant_ngrams)
    remaining_tagged = nltk.pos_tag([i for i in remaining_ngrams if i])

    shuffle(remaining_tagged)

    random_ngrams = []
    for _, tag in relevant_tagged:
        random_pos_tag = next(pos_tag for pos_tag in remaining_tagged
                              if pos_tag[1] == tag)
        remaining_tagged.remove(random_pos_tag)
        random_ngrams.append(random_pos_tag[0])
    print(random_ngrams)


def control_handler(n, number_of_ngrams):
    output_directory = f"{DIR_PATH}/../output/"

    for filename in os.listdir(output_directory):
        if not filename.startswith(f"{n}-gram"):
            continue

        print(filename)
        ngrams = []
        with open(os.path.join(output_directory, filename)) as f:
            for line in f.read().splitlines():
                if line.lower().startswith("position\t"):
                    continue

                _, ngram = line.split("\t")[1:]
                ngrams.append(ngram)

        create_control(ngrams, number_of_ngrams)


if __name__ == "__main__":
    control_handler(1, 20)
