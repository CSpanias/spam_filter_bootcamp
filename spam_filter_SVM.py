import pandas as pd
import numpy as np

from nltk import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# set output to be auto-adjusted relative to terminal size
pd.options.display.width = 0

# import csv file
df = pd.read_csv(r"C:\Users\10inm\Desktop\spam_filter_bootcamp\spam.csv",
                 encoding='latin-1')

""" 
1. DATA PRE-PROCESSING
"""
# check dataset
print(df.info())
# check for missing values
print(df.isna().sum())
# remove unwanted columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1,
        inplace=True)
# give columns a more descriptive names
df.rename({'v1': 'labels', 'v2': 'message'}, axis=1, inplace=True)
# check how many data points we have in each category
print(df['labels'].value_counts())
# convert target to numerical
df['label'] = df['labels'].map({'ham': 0, 'spam': 1})
# drop original labels columns
df.drop(['labels'], axis=1, inplace=True)
# confirm how many data points we have in each category
print(df['label'].value_counts())
# check that everything looks OK
print(df.head())
print(df.info())


# now loop over each sentence and tokenize it separately
df['tokenized_sents'] = df.apply(lambda row: word_tokenize(row['message']),
                                 axis=1)



"""
2. BUILD THE MODEL
"""
# # assign feature (X) and target (y) variables and convert them to arrays
X = df[['message']]
y = df['label'].values
# # check that X = 2-Dimensional and y = 1-Dimensional
#  print(X.shape)
#  print(y.shape)
# # split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                     shuffle=True, random_state=1)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words="english")
vect.fit(X_train) # Find some word that cause most of spam email
print(X_train)
# # select the model
# model = SVC()
# # train the model
# model.fit(X_train, y_train)
# # make predictions with the test data
# y_pred = model.predict(X_test)
# # check accuracy of the model
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


