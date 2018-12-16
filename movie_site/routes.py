from flask import render_template, url_for, flash, redirect, request
from movie_site import app, db
import numpy as np
import re
import os
import pickle
from vectorizer import vect


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title='Home')


@app.route("/home", methods=['POST'])
def predict():

    # Unpickle the classifier
    cur_dir = os.path.dirname(__file__)
    clf = pickle.load(open(
        os.path.join(cur_dir,
                     'pkl_objects',
                     'classifier.pkl'), 'rb'))
    if request.method == 'POST':
        # Label to display
        label = {0: 'Negative', 1: 'Positive'}
        # Get the review from the user
        example = [request.form['comment']]

        # sentiment analysis on the input
        X = vect.transform(example)
        prediction = ('Prediction: %s\n' % (label[clf.predict(X)[0]]))
        probability = ('Probability: %.2f%%' %
                       (np.max(clf.predict_proba(X))*100))
    return render_template("response.html", prediction=prediction,
                           probability=probability)
