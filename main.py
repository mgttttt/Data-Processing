import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

directory = "data/movie/neg/"
files = os.listdir(directory)
lst_pos = []
lst_neg = []
ones_zeros = []

for file in files:
    ones_zeros.append(0)
    f = open(directory + file, 'r')
    lst_neg.append(f.read())

directory = "data/movie/pos/"
files = os.listdir(directory)

for file in files:
    ones_zeros.append(1)
    f = open(directory + file, 'r')
    lst_pos.append(f.read())

lst_of_all = lst_neg + lst_pos

words_set = set()

for el in lst_neg:
    for word in el.split():
        words_set.add(word)

for el in lst_pos:
    for word in el.split():
        words_set.add(word)

result = np.zeros((len(lst_of_all), len(words_set)))
words_set = list(words_set)
print(result.shape)
for i, sequence in enumerate(lst_of_all):
    for j in sequence.split():
        result[i, words_set.index(j)] = 1

X_train, X_test, Y_train, Y_test = train_test_split(result, ones_zeros, test_size=0.2, random_state=0)

clf = BernoulliNB()

clf.fit(X_train, Y_train)

print(metrics.accuracy_score(clf.predict(X_test), Y_test))

