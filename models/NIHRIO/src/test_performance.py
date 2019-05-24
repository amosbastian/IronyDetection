import logging
import os

import numpy as np

from DataProcessor import DataProcessor
from MLP import MLP

logging.basicConfig(level=logging.INFO)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
n_fold = 10
train_file = "data/SemEval2018-T3-taskA.txt"
test_file = "data/SemEval2018-T3_input_test_taskA.txt"
training_directory = "{}/../../../datasets/train/".format(DIR_PATH)
predictions_directory = "{}/predictions".format(DIR_PATH)


def test_performance():
    predictions = [filename.replace("predictions_", "") for filename in
                   os.listdir(predictions_directory)]
    print(predictions)
    for filename in os.listdir(training_directory):
        if "README" in filename or "CONTROL" in filename or filename in predictions:
            continue
        logging.info("Training with dataset: {}".format(filename))

        train_file = "{}{}".format(training_directory, filename)
        train_data, test_data = DataProcessor().process_data(
            train_file, test_file, load_saved_data=False)

        k_fold_train, k_fold_valid = DataProcessor.split_kfolds(
            train_data, n_fold)

        mlp_predict = None
        mlp_f1_scores = []

        for i in range(len(k_fold_train)):
            print("====================Fold %d=================" % (i + 1))
            _, _, mlp_pred_test, mlp_f1_score = MLP().predict(
                k_fold_train[i], k_fold_valid[i], test_data)
            mlp_f1_scores.append(mlp_f1_score)
            if mlp_predict is None:
                mlp_predict = mlp_pred_test
            else:
                mlp_predict = np.column_stack((mlp_predict, mlp_pred_test))

        p_filename = "{}/predictions/predictions_{}".format(DIR_PATH, filename)
        PREDICTIONSFILE = open(p_filename, "w")

        mlp_predict = np.average(mlp_predict, axis=1)
        for i in range(len(mlp_predict)):
            if i > 0:
                label = mlp_predict[i]
                if label > 0.5:
                    label = 1
                else:
                    label = 0
                PREDICTIONSFILE.write("%d\n" % label)

        PREDICTIONSFILE.close()
        mlp_f1_scores = np.array(mlp_f1_scores)

        file_out = open("{}/output/output.csv".format(DIR_PATH), "a")
        f1_score = "{},{}".format(mlp_f1_scores.mean(),
                                  mlp_f1_scores.std() * 2)
        file_out.write("{},{}\n".format(filename, f1_score))
        print("Final mlp F1: {}".format(f1_score))
        file_out.close()


if __name__ == "__main__":
    test_performance()
