__author__ = 'denis'


class MultinomialNaiveBayes(object):
    def __init__(self, alpha = 1.0):
        # smoothness parameter
        self.__alpha = alpha

    def load_model(self, file_name):
        pass

    def save_model(self, file_name):
        pass

    def train(rows, answers):
        pass

    def predict(rows):
        pass

    def __str__(self):
        description = "Here is a description of NB and some statistic..."
        return description


