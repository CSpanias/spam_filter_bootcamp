"""
Spam Detection Notebook by Farid Muradov:
https://www.kaggle.com/zkop354/spam-detection
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# set output to be auto-adjusted relative to terminal size
pd.options.display.width = 0

# import csv file
df = pd.read_csv(r"C:\Users\10inm\Desktop\spam_filter_bootcamp\emails.csv",
                 encoding='latin-1')
print(df.spam.value_counts())

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(df["text"], df["spam"],
                                                    test_size=0.2, random_state=10)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words="english")
# Find some word that cause most of the spam email
vect.fit(train_X)
print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])

X_train_df = vect.transform(train_X)
X_test_df = vect.transform(test_X)
type(X_test_df)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
model = MultinomialNB(alpha=1.8)
model.fit(X_train_df, train_y)
pred = model.predict(X_test_df)
print(accuracy_score(test_y, pred))

print(classification_report(test_y, pred, target_names=["Not Spam", "Spam"]))
print(confusion_matrix(test_y, pred))

# non-spam email
print(df["text"][1472])
pred = model.predict(vect.transform(df["text"]))
print("Pred : ", pred[1472])
print("Main : ", df["spam"][1472])

# spam email
print(df["text"][10])
pred = model.predict(vect.transform(df["text"]))
print("Pred : ", pred[10])
print("Main : ", df["spam"][10])