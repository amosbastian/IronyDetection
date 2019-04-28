import os


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


if __name__ == "__main__":
    print(parse_dataset("SemEval2018-T3-train-taskA"))
