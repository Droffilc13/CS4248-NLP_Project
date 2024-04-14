import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from removal_methods.basic_removal import basicRemove
from removal_methods.zipfMethod import removeHighTF, removeLowTF, removeTF1
from removal_methods.tbrs import tbrs
from removal_methods.mutualinfo import mutualInfo

from utils.utils import save, generateCount, loadCounter, plotZipf


'''
This file is to serve two purposes of cleaning of the data:
    1) Remove new lines, tabs, links, special characters (except for [!, ? and - connecting 2 words (e.g. Wi-Fi)])
    2) Dynamically generate stopword list
        a) By removing High TF
        b) By removing Low TF
        c) By removing term with frequencies of 1
        d) By conducting term-based random sampling
        e) by removing words with low mutual information
'''

def main():
    data = pd.read_csv("../raw_data/fulltrain.csv", names=["label", "text"])
    test = pd.read_csv("../raw_data/balancedtest.csv", names=['label', 'text'])

    # Filter for data with labels only in [1, 2, 3, 4]
    data = data.loc[np.in1d(data['label'], [1, 2, 3, 4])]
    test = test.loc[np.in1d(test['label'], [1, 2, 3, 4])]

    x = data['text']
    y = data['label']

    '''
    We originally ran these 2 lines to conduct basic removal on the text data and stored it to the file,
    'basic_removal.csv' to be more efficient in our development and avoid rerunning long processes by 
    loading cleaned data.

    # clean_x = basicRemove(x)
    # save(clean_x, 'basic_removal.csv')
    
    We did the same thing for generating a correct term count for the cleaned data.

    # generateCount(clean_x, 'wordCount.txt')
    '''

    clean_x = pd.read_csv("./basic_removal.csv").squeeze("columns")
    counter = loadCounter("wordCount.txt")

    stoplist, removed_x = removeTF1(clean_x, counter)

    '''
    Similar to before, we saved the result of basic cleaning the test data as well.

    # clean_x_test = basicRemove(test['text'])
    # save(clean_x_test, 'test_basic_removal.csv')
    '''
    clean_x_test = pd.read_csv("./test_basic_removal.csv").squeeze("columns")


    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter= 5000)

    print('Training model')
    model.fit(vectorizer.fit_transform(removed_x), y)
    print('Predicting based on model')
    y_pred = model.predict(vectorizer.transform(removeTF1(clean_x_test, loadCounter('testWordCount.txt'))))

    score = f1_score(test['label'], y_pred, average='macro')
    print('score on validation = {}'.format(score))
    
    report = classification_report(test['label'], y_pred, target_names=["1", "2", "3", "4"])
    print(report)

    plotZipf(counter, 500)

if __name__ == '__main__':
    main()
