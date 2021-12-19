#!/usr/bin/env python

__author__ = "Daniel Ene"
__credits__ = ["Daniel Ene", "Saad Khan"]
__license__ = "GPL"
__version__ = "1.0"

import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm, kernel_approximation, linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import numpy as np
import re
import nltk

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_choice', default=3, type=int, help='1 = SVM, 2 =  Linear kernel SVM, 3 = SGDClassifier')
parser.add_argument('-v', '--vectorizer', default=2, type=int, help='1 = CountVectorizer, 2 =  TfidfVectorizer')
args = parser.parse_args()

DATASET_PATH = "emails.csv"

#Step 1: Load, explore and preprocess dataset
# Parts of function sanitise were sourced from **Chaitanya Baranwal** (https://github.com/chaitanyabaranwal/)
# This does not work as intended, requires a review.
def sanitise(email):
    # convert text to lowercase
    email = email.lower()

    # convert URLs to 'httpaddr'
    email = re.sub(r'(http|https)://[^\s]*', r' httpaddr ', email)

    # convert email addresses to 'emailaddr'
    email = re.sub(r'[^\s]+@[^\s]+[.][^\s]+', r' emailaddr ', email)

    # convert numbers to 'number'
    email = re.sub(r'[0-9]+', r' number ', email)

    # convert $, ! and ? to proper words
    email = re.sub(r'[$]', r' dollar ', email)
    email = re.sub(r'[!]', r' exclammark ', email)
    email = re.sub(r'[?]', r' questmark ', email)

    # convert other punctuation to whitespace
    email = re.sub(r'([^\w\s]+)|([_-]+)', r' ', email)

    # convert newlines and blanklines to special strings and extra whitespace to single
    email = re.sub(r'\n', r' newline ', email)
    email = re.sub(r'\n\n', r' blankline ', email)
    email = re.sub(r'\s+', r' ', email)
    email = email.strip(' ')

    # perform word stemming
    emailWords = email.split(' ')
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    stemWords = [stemmer.stem(word) for word in emailWords]
    email = ' '.join(stemWords)
    
    return email

df = pd.read_csv(DATASET_PATH)
#print(df.head(10), end="\n")
print(df.describe, end="\n\n")

# df["EmailText"].apply(sanitise)
# x.to_csv("spam_correct.csv", index=False)

x = df["EmailText"]
y = df["Label"]

# print("I found missing values at: " + str(np.where(pd.isnull(y))))
# y.replace(np.nan, 1) #there are no missing labels
print("There are " + str(y.nunique()) + " unique values in label column. You should have two..." , end="\n\n")


#Step 2: Split Dataset 75-25
#split = int(len(df) * 0.75)
#x_train, y_train = x[:split], y[:split]
#x_test, y_test = x[split:], y[split:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=42)

#Step 3: Extract Features
cv = CountVectorizer()  if args.vectorizer == 1 else TfidfVectorizer()
features = cv.fit_transform(x_train)
#print(features[0])
# all emails, all unique words become columns each with its own datapoint - sparse data (50k freqs for each email, lots of 0s), perfect for PCA 

#Step 4: Build and train models and tune hyperparameters
'''
1: SVC - SVM classifier
2: LinearSVC - SVC with parameter kernel='linear', but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples. The combination of penalty='l1' and loss='hinge' is not supported. 
3: SGDClassifier - implements regularized linear models with stochastic gradient descent (SGD) learning. 'elasticnet' loss might bring sparsity to the model not achievable with 'l2'. Also has hyperpar 'class_weight' but I'm not sure how to implement it so I can make model more sensitive to spam.
'''
#tuned_parameters1 = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [1e-3, 
#                    1e-4],'C': [1, 100, 1000]} # Best parameters were found to be the ones selected below
tuned_parameters1 = {'kernel': ['rbf'], 'gamma': [1e-2],'C': [100]}
tuned_parameters2 = {'C': [10000], 'max_iter':[50000]}
tuned_parameters3 = {'penalty': ['l1', 'l2', 'elasticnet'], 'alpha': [0.00001, 0.0001, 
                    0.001, 0.01], 'max_iter':[500, 1000, 2000], 'learning_rate': 
                    ['constant', 'optimal', 'invscaling', 'adaptive'], 'early_stopping':[True]}

model1 = GridSearchCV(svm.SVC(), tuned_parameters1)
model2 = GridSearchCV(svm.LinearSVC(), tuned_parameters2)
model3 = GridSearchCV(linear_model.SGDClassifier(), tuned_parameters3)

if args.model_choice == 1:
    model = model1
elif args.model_choice == 2:
    model = model2
else:
    model = model3

model.fit(features, y_train)
print("Train accuracy is: " + str((model.score(cv.transform(x_train), y_train)) * 100) + "%", end="\n")
print("For " + str(model.estimator) + " I found the best hyperparameters to be: " + str(model.best_params_), end="\n\n")


#Step 5: Evaluation (Learn more about the classif report: https://medium.com/@kennymiyasato/classification-report-precision-recall-f1-score-accuracy-16a245a437a5)
y_pred = model.predict(cv.transform(x_test))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 

print("Test accuracy is: " + str((model.score(cv.transform(x_test), y_test)) * 100) + "%")


