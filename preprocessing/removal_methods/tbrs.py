import numpy as np

from collections import Counter

'''
    Normalize count values in counter using (x - x_min + .01) / (x_max - x_min), the numerator is added with .01 to avoid zero
'''

def normalizeCounter(counter):

    x_max = (max(list(counter.values())))
    x_min = (min(list(counter.values())))

    normalized_counter = {k: ((v - x_min + .01) / (x_max - x_min)) for k, v in counter.items()}

    return normalized_counter

'''
    Calculate Kuller-Leibler (KL) Divergence for each randomly chosen chunk and return term with 
    lowest KL divergence
'''

def minKLDivergence(line, counter):
    
    lineTF = dict(Counter(line.lower().split(" ")))
    res = ""
    min = 10000

    for k, v in lineTF.items():
        val = v * np.log(v / counter[k])
        if val < min and k not in ['?', '!']:
            res = k
            min = val

    return res


'''
    Randomly select chunks (entries) to analyse for KL Divergence and build stoplist
'''

def tbrs(data, counter, frac = 0.75):
    print("Doing TBRS")
    random_sample = data.sample(frac = frac)
    stoplist = set(())

    normalized_counter = normalizeCounter(counter)

    for i in random_sample:
        stoplist.add(minKLDivergence(i, normalized_counter))

    print("Length of TBRS stoplist is {}".format(len(stoplist)))
    print("Done TBRS")
    return stoplist, data.apply(lambda line: ' '.join([word for word in line.split(" ") if word.lower() not in stoplist]))