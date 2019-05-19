import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
base_filename = "predictions_SemEval2018-T3-train-taskA_emoji"


def get_predictions(filename):
    """Returns the predictions (list of 0s and 1s) from the given filename."""
    with open(f"{DIR_PATH}/predictions/{base_filename}{filename}") as f:
        predictions = [int(x) for x in f.readlines()]
    return predictions


def get_tweets(filename):
    """Gets the tweets from the training set with the given filename."""
    training_filename = base_filename.replace('predictions_', '')
    training_filename = f"{training_filename}{filename}"

    tweets = []
    with open(f"{DIR_PATH}/../../datasets/train/{training_filename}") as f:
        for line in f.readlines():
            if line.lower().startswith("tweet index"):
                continue

            line = line.rstrip()
            tweet = line.split("\t")[2]
            tweets.append(tweet)

    return tweets


def main(filename):
    # Get predictions
    predictions = get_predictions(filename)
    base_predictions = get_predictions(".txt")

    # Get indices of predictions that were first predicted as ironic, but are
    # now predicted as non-ironic
    sarcastic_changes = [i for i, prediction in enumerate(predictions)
                         if prediction == 0 and base_predictions[i] == 1]

    # Get the tweets
    tweets = get_tweets(filename)
    base_tweets = get_tweets(".txt")

    print(f"{base_filename}{filename} caused the following tweets that were "
          "previously classified as ironic, to be classified as non-ironic:")

    for i in sarcastic_changes:
        base_tweet = base_tweets[i]
        new_tweet = tweets[i]

        if base_tweet != new_tweet:
            print(f"{base_tweet} -> {new_tweet}")


if __name__ == "__main__":
    main("_1-gram_frequency_ironic_1.txt")