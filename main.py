__author__ = 'denis'

import model.naive_bayes as nb


def main():

    try:
        print "Naive Bayes."

        model = nb.MultinomialNaiveBayes()
        print model

        


    except Exception, err:
        print "Error: ", str(err)




if __name__ == "__main__":

    main()
