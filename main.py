__author__ = 'denis'

import json

import model.naive_bayes as nb


def split_dataset(data_rows, data_ans, train_portion=0.7):
    train_rows, test_rows = \
        data_rows[:int(train_portion * len(data_rows))], data_rows[int(train_portion * len(data_rows)):]

    train_ans, test_ans = \
        data_ans[:int(train_portion * len(data_ans))], data_ans[int(train_portion * len(data_ans)):]

    return train_rows, test_rows, train_ans, test_ans


def main():
    try:
        data_storage_name = "mini_data.json"


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
        predicted_answers = model.predict(test_rows)

        for p_answer,t_answer in zip(predicted_answers, test_ans):
            print "Predicted answer: ", p_answer[0]
            print "True ground answer: ", t_answer



    except Exception, err:
        print "Error: ", str(err)


if __name__ == "__main__":

    main()
