__author__ = 'denis'

import operator


def compute_prior(current_class_frequency, total_classes):
    return float(current_class_frequency) / float(total_classes)


def count_w_appear_in_text_c(word, class_name, dataset_rows, dataset_answers):
    # TODO refactor this func later

    count_w_aitc = 0  # how many times word appears in text with class C
    count_w_itc = 0  # how many words in text with class C
    for index in range(len(dataset_answers)):
        if dataset_answers[index] == class_name:
            count_w_aitc += dataset_rows[index].count(word)
            count_w_itc += len(dataset_rows[index])

    return count_w_aitc, count_w_itc


class MultinomialNaiveBayes(object):
    def __init__(self, alpha=1.0):
        self.__alpha = alpha  # smoothness parameter
        self.__vocabulary_words = []  # All unique words in a training text
        self.__vocabulary_classes = []  # All classes in a training text
        self.__classes_frequency = {}
        self.__priors = {}
        self.__likelihoods = {}

        self.__unknown_word = "unknown"  # Label for word that haven't been in training set

    def train(self, rows, answers):
        # TODO refactor this method after...

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

        vocabulary_volume = len(self.__vocabulary_words)

        for class_name in self.__vocabulary_classes:

            likelihood_per_class = {}

            for word in self.__vocabulary_words:
                count_w_aitc, count_w_itc = count_w_appear_in_text_c(word, class_name, rows, answers)

                value = float(count_w_aitc + self.__alpha) / float(count_w_itc + self.__alpha * vocabulary_volume)

                likelihood_per_class.update({word: value})

            # Add extra column for unknown word
            count_w_aitc, count_w_itc = count_w_appear_in_text_c(self.__unknown_word, class_name, rows, answers)
            value = float(0.0 + self.__alpha) / float(count_w_itc + self.__alpha * vocabulary_volume)
            likelihood_per_class.update({self.__unknown_word: value})

            self.__likelihoods.update({class_name: likelihood_per_class})

            # just a little bit test
            # print(self.__likelihoods["italian"]["sea salt"])


    def predict(self, rows):

        answer_list = []

        for doc_for_test in rows:
            argmax_c = {}

            for class_i in self.__vocabulary_classes:
                estimated_likelihood = 1.0
                for word in doc_for_test:
                    if self.__likelihoods[class_i].get(word, self.__unknown_word) != self.__unknown_word:
                        estimated_likelihood *= self.__likelihoods[class_i][word]
                    else:
                        estimated_likelihood *= self.__likelihoods[class_i][self.__unknown_word]

                estimated_value = self.__priors[class_i] * estimated_likelihood
                argmax_c.update({class_i: estimated_value})

            answer_list.append(max(argmax_c.iteritems(), key=operator.itemgetter(1)))

        return answer_list


    def load_model(self, file_name):
        pass


    def save_model(self, file_name):
        pass


    def __str__(self):
        description = "Here is a description of NB and some statistic..."
        return description


