from emoji_statistics import emoji_statistics
from utils import (irony_comparison_handler, ngram_removal_handler,
                   parse_dataset, tokenise_default)
from word_statistics import word_statistics

if __name__ == "__main__":
    labels, corpus = parse_dataset("SemEval2018-T3-train-taskA_emoji")
    word_statistics(labels, corpus)
    emoji_statistics(labels, corpus)
    irony_comparison_handler()
    tokenise_default()
    ngram_removal_handler("SemEval2018-T3-train-taskA_emoji", range(1, 31))
