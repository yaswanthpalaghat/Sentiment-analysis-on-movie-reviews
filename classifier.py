# License: MIT
# https://github.com/rasbt/python-machine-learning-book/blob/master/LICENSE.txt

import numpy as np
import re
import nltk
import pickle
import os
from nltk.corpus import stopwords
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# Using sci-kit learn SGDClassifier with logistic regression
if Version(sklearn_version) < '0.18':
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
else:
    clf = SGDClassifier(loss='log', random_state=1, max_iter=1)

# Define stopwords to remove from the 'bag of words'
stop = stopwords.words('english')

# Grab each line of CSV individually
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# Dataset for the analysis
doc_stream = stream_docs(path='movie_data.csv')

# Get the minibatch of multiple reviews
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

# tokenize the text i.e. clean
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Create the bag of words
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

# train the model
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000) # Get the training set
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)

X_test, y_test = get_minibatch(doc_stream, size=5000) # Get the test set
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

# pickle the stopwords and classifier for use in the web app
pickle.dump(stop,
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol=4)

pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
