import re
import json
import matplotlib.pyplot as plt
from collections import Counter

'''
    Save file to .csv
'''
def save(file, filename):
    file.to_csv(filename, index=False)

'''
    Generate the term counts in a Dataframe, with an option to save it to a .txt file
'''
def generateCount(x, fp = None):
    counter = Counter()
    for i in range(len(x)):
        line = x[i]
        counter += Counter(line.lower().split(" "))
        print("Generating count progress: {:.2f}%".format((i + 1)/len(x) * 100))

    if fp:
        with open(fp, 'w') as file:
            file.write(str(dict(counter)))

    return counter

'''
    Load term counts from a .txt file indicated by filepath
'''
def loadCounter(filepath):
    count = ""
    with open(filepath) as file:
        count = file.read()
        count = re.sub("\'", "\"", count)

    count = json.loads(count)

    count = dict(sorted(count.items(), key = lambda item: item[1], reverse = True))

    return count

'''
    Given the term counts, plot a Zipf graph to visualize and utilize the elbow method
'''
def plotZipf(counter: Counter, threshold = None):
    
    if not threshold:
        threshold = len(counter.values())

    plt.plot(list(range(1, threshold + 1)), list(counter.values())[:threshold])
    plt.axvline(x=6, color='r')
    plt.axvline(x=100, color = 'r')
    plt.show()
