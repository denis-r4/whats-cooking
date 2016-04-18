# whats-cooking
    It's a straightforward implementation of "What's cooking" task from Kaggle (kaggle.com/c/whats-cooking)
    based on a multinomial naive bayes classifier. Source dataset contains 20 classes,
    simple test shows overall accuracy - 73.03%, but it's not very informative accuracy metric
    without precision, recall and F.
    
    Requirements: See https://github.com/denis-r4/whats-cooking/blob/master/requirements.txt
    Usage: python main.py
    Note: source_data.json should be in the same folder as the main.py

    Here is still a lot of work:
    1) save/load model
    2) accuracy estimation metrics (precision, recall, F-measure)
    3) refactor some code
    4) add two mods of usage main script: --predict and --train
