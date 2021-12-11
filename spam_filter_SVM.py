"""
This code is based on the YouTube tutorial::
	Email Spam Detection Using Python & Machine Learning:
	https://www.youtube.com/watch?v=cNLPt02RwF0&list=WL&index=4&t=12s

It was modified based on:
	"Spam or Ham" tutorial:
	https://github.com/tejank10/Spam-or-Ham/blob/master/spam_ham.ipynb

	Sololearn (Machine Learning path):
	https://www.sololearn.com/learning/1094

	Machine Learning for Absolute Beginners book:
	https://www.amazon.co.uk/Machine-Learning-Absolute-Beginners-Introduction-ebook/dp/B08RWBSKQB
"""
import pandas as pd
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from wordcloud import WordCloud

# read the CSV file
df = pd.read_csv(r"C:\Users\10inm\Desktop\spam_filter_bootcamp\emails.csv")
# print the first 5 rows of data
print(df.head())
# print the number of rows and columns
print(df.shape)

"""
DATA CLEANING PROCESS
"""

# check for duplicates
print(df.duplicated().sum())
# remove duplicates
print(df.drop_duplicates(inplace=True))
# check for missing values
print(df.isna().sum())

"""
Define a function that:
	1. Removes punctuation symbols and stopwords from a text
	2. Splits the text into individual words
	3. Stores the individual words in a list
"""
def process_text(text):
	"""
		Returns a list of strings (individual words) of a text (paragraph) excluding
		punctuation, stopwords.

			Parameters
			----------
				text: An object (string) variable.

			Returns
			-------
				clean_words: A list of objects (string).
		"""
	# create an empty list to later store the rows (emails) without punctuation
	no_punctuation = []
	# create an empty list to later store the words of each row (email)
	clean_words = []
	# for every row (email) in the text column
	for char in text:
		# remove every punctuation symbol
		if char not in string.punctuation:
			# append the free-of-punctuation row in the list "no_punctuation"
			no_punctuation.append(char)
	# insert an empty space between words
	no_punctuation = ''.join(no_punctuation)

	# for every row (email without punctuation symbols) in
	# the "no_punctuation" list split the text into words
	for word in no_punctuation.split():
		# convert every word in lowercase
		word = word.lower()
		# if this word is not a stopword
		if word not in stopwords.words('english'):
			# append the free-of-stopwords words in the list "clean_words"
			clean_words.append(word)
	# give back the list "clean_words"
	return clean_words


# apply the "process_text" function to the column "text"
# so we end up with a column of individual words
df['text'].apply(process_text)
# print the first 5 rows of the column 'text'
print(df['text'].head())
# create a matrix of words based on their frequency
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])
"""
CountVectorizer transforms a given text into a vector on the basis of 
the frequency (count) of each word that occurs in the entire text. 
(*bow = bog of words)
More info on how it works: 
https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/
"""

"""
MODEL TRAINING PROCESS
"""
# assign X and y variables
X = messages_bow
y = df['spam']
# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=1)
# get the shape of X (rows, columns)
print(X.shape)
# select the model (Naive Bayes Classifier)
model = SVC(kernel='rbf', C=100, gamma='auto')
# train the model
model.fit(X_train, y_train)

"""
EVALUATION PROCESS
"""

# check the prediction on the training set
y_pred_train = model.predict(X_train)
print('Accuracy Score on Training set: ',
      round(accuracy_score(y_train, y_pred_train), 4))
# use the model to predict on the testing set
y_pred_test = model.predict(X_test)
print('Accuracy Score on Testing set: ',
      round(accuracy_score(y_test, y_pred_test), 4))

# print the classification report on the testing set
print("\n\t\t\t\t\tCLASSIFICATION REPORT\n\n",
      classification_report(y_test, y_pred_test))

# print the confusion matrix of the testing set
print("\nCONFUSION MATRIX\n", confusion_matrix(y_test, y_pred_test))
