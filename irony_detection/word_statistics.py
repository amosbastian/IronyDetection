import os
import string

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


def parse_dataset(training_set):
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


def word_frequency(corpus):
    words = []
    for sentence in corpus:
        # Remove punctuation
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))

        # Extend words list while removing stop words
        words.extend([word for word in text_processor.pre_process_doc(sentence)
                      if word not in stopwords.words("english")])

    print(FreqDist(words).most_common(20))

if __name__ == "__main__":
    labels, corpus = parse_dataset("SemEval2018-T3-train-taskA")
    word_frequency(corpus)
