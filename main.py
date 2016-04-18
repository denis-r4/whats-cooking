__author__ = 'denis'

import json
import model.naive_bayes as nb
import csv

def split_dataset(data_rows, data_ans, train_portion=0.7):
    train_rows, test_rows = \
        data_rows[:int(train_portion * len(data_rows))], data_rows[int(train_portion * len(data_rows)):]

    train_ans, test_ans = \
        data_ans[:int(train_portion * len(data_ans))], data_ans[int(train_portion * len(data_ans)):]

    return train_rows, test_rows, train_ans, test_ans


def main():
    try:

        data_storage_name = "source_data.json"

        # Extracting and splitting the data
        with open(data_storage_name) as data:
            json_data = json.load(data)

        dataset_size = len(json_data[:])
        dataset_rows = []
        dataset_answers = []

        for i in range(dataset_size):
            dataset_rows.append(json_data[i]["ingredients"])
            dataset_answers.append(json_data[i]["cuisine"])

        train_rows, test_rows, train_ans, test_ans = split_dataset(dataset_rows, dataset_answers)

        model = nb.MultinomialNaiveBayes()
        print model

        model.train(train_rows, train_ans)

         # TODO there is an issue, test data shouldn't contain a class, that didn't appear on training stage.
        predicted_answers = model.predict(test_rows)
        if predicted_answers is None:
            raise ValueError('Classifier data missing. Train or load classifier before predict')

        numerator = 0.0
        denominator = 0.0
        for p_answer, t_answer in zip(predicted_answers, test_ans):
            print "Predicted answer: ", p_answer[0]
            print "True ground answer: ", t_answer
            denominator += 1
            if p_answer[0] == t_answer:
                numerator += 1

        print "Overall accuracy: ", numerator / denominator


    except Exception, err:
        print "Error: ", str(err)


if __name__ == "__main__":
    main()

    """
    It's a straightforward implementation of "What's cooking" task from Kaggle (kaggle.com/c/whats-cooking)
    based on a multinomial naive bayes classifier. Source dataset contains 20 classes,
    simple test shows overall accuracy - 73.03%, but it's not very informative accuracy metric
    without precision, recall and F.

    Usage: python main.py
    source_data.json should be in the same folder as the main.py

    Here is still a lot of work:
    1) save/load model
    2) accuracy estimation
    3) refactor some code
    4) add two mods of usage main script: --predict and --train
    """
