import pandas as pd
import numpy as np

from collections import Counter
from utils.utils import generateCount, loadCounter

'''
    Calculate probability of terms in a dataframe
'''

def getTermProbs(counter):
    totalCount = sum(list(counter.values()))
    res = {k: v / totalCount for k, v in counter.items()}

    return res

'''
    Calculate Mutual Information
'''
def mutualInfo(x: pd.Series, y: pd.Series, counter):
    data = pd.concat((x, y), axis = 1)

    class_1 = data[data['label'] == 1].reset_index(drop = True)
    class_2 = data[data['label'] == 2].reset_index(drop = True)
    class_3 = data[data['label'] == 3].reset_index(drop = True)
    class_4 = data[data['label'] == 4].reset_index(drop = True)

    '''
    Similar to cleaning.py, we generated the term counts for each classes and saved them to the respective
    .txt files to improve efficiency.

    # termProb_1 = getTermProbs(generateCount(class_1['text'], 'class_1.txt'))
    # termProb_2 = getTermProbs(generateCount(class_2['text'], 'class_2.txt'))
    # termProb_3 = getTermProbs(generateCount(class_3['text'], 'class_3.txt'))
    # termProb_4 = getTermProbs(generateCount(class_4['text'], 'class_4.txt'))
    '''

    termProb_1 = getTermProbs(loadCounter('class_1.txt'))
    termProb_2 = getTermProbs(loadCounter('class_2.txt'))
    termProb_3 = getTermProbs(loadCounter('class_3.txt'))
    termProb_4 = getTermProbs(loadCounter('class_4.txt'))

    termProb_x = getTermProbs(counter)

    # calculating mutual information
    mi_1 = {k: v * np.log(v / (termProb_x[k] * len(class_1) / len(x))) for k, v in termProb_1.items()}
    mi_2 = {k: v * np.log(v / (termProb_x[k] * len(class_2) / len(x))) for k, v in termProb_2.items()}
    mi_3 = {k: v * np.log(v / (termProb_x[k] * len(class_3) / len(x))) for k, v in termProb_3.items()}
    mi_4 = {k: v * np.log(v / (termProb_x[k] * len(class_4) / len(x))) for k, v in termProb_4.items()}
    
    # merging the mutual informations by simple addition
    mi = dict(list(mi_1.items()) + 
              [(k, mi_1[k] + mi_2[k]) if k in mi_2 else (k, mi_1[k]) for k in mi_1.keys()] +
              [(k, mi_2[k]) for k in mi_2 if k not in mi_1])
    mi.update({k: mi[k] + mi_3[k] if k in mi_3 else mi[k] for k in mi.keys()})
    mi.update({k: mi_3[k] for k in mi_3 if k not in mi})
    mi.update({k: mi[k] + mi_4[k] if k in mi_4 else mi[k] for k in mi.keys()})
    mi.update({k: mi_4[k] for k in mi_4 if k not in mi})

    # sort by ascendingmutual information
    res = [k for k, _ in sorted(mi.items(), key = lambda item: item[1], reverse=False)]

    stoplist = res[-70:]

    return stoplist, x.apply(lambda line: ' '.join([word for word in line.split(" ") if word.lower() not in stoplist]))