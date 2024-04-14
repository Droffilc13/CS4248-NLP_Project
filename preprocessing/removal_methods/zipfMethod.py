import pandas as pd
from collections import Counter

'''
    Remove terms with high term frequencies as visualized by Zipf's graph
'''
def removeHighTF(x: pd.Series, counter: Counter, threshold = 6):
    stopword_list = list(counter.keys())[:threshold]

    return stopword_list, x.apply(lambda line: ' '.join([word for word in line.split(" ") if word.lower() not in stopword_list]))

'''
    Remove terms with low term frequencies as visualized by Zipf's graph
'''
def removeLowTF(x: pd.Series, counter: Counter, threshold = 100):
    whitelist = list(counter.keys())[:threshold]

    return list(counter.keys())[threshold:], x.apply(lambda line: ' '.join([word for word in line.split(" ") if word.lower() in whitelist]))

'''
    Remove terms that only appear once
'''
def removeTF1(x: pd.Series, counter: Counter):
    return list(dict(filter(lambda item: item[1] == 1, counter.items())).keys()), x.apply(lambda line: ' '.join([word for word in line.split(" ") if counter[word.lower()] > 1]))

