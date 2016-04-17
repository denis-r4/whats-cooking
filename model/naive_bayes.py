__author__ = 'denis'

def compute_prior(current_class_frequency, total_classes):
    return float(current_class_frequency) / float(total_classes)



class MultinomialNaiveBayes(object):
    def __init__(self, alpha=1.0):
        self.__alpha = alpha  # smoothness parameter
        self.__vocabulary_words = []  # All unique words in a training text
        self.__vocabulary_classes = []  # All classes in a training text
        self.__classes_frequency = {}
        self.__priors = {}

    def train(self, rows, answers):
        # TODO refactor this method before...

        # 1. Extract vocabulary (unique words) and classes names and their frequencies.
        for row in rows:
            for word in row:
                if not (word in self.__vocabulary_words):  # if (vocabulary.count(word_j) == 0):
                    self.__vocabulary_words.append(word)

        # print "Vocabulary: ", vocabulary_words
        print "Vocabulary size: ", len(self.__vocabulary_words)

        for word in answers:
            if not (word in self.__vocabulary_classes):
                self.__vocabulary_classes.append(word)

        for word in self.__vocabulary_classes:
            frequency = answers.count(word)
            self.__classes_frequency.update({word: frequency})

        # print "Classes frequency: ", classes_frequency
        print "Amount of classes: ", len(self.__classes_frequency)



        # 2. Compute priors
        dataset_size = len(answers)

        for class_name in answers:
            frequency = self.__classes_frequency[class_name]
            prior = compute_prior(frequency, dataset_size)
            self.__priors.update({class_name: prior})

        # print "Priors: ", priors
        print "Priors size: ", len(self.__priors)



        # 3. Compute likelihoods


    def predict(self, rows):
        pass


    def load_model(self, file_name):
        pass

    def save_model(self, file_name):
        pass

    def __str__(self):
        description = "Here is a description of NB and some statistic..."
        return description


