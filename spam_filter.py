# Spam Filter Project for AI & ML Bootcamp

# Detects if an email is spam (1) or not (0).

# Import libraries
import pandas as pd
import string

from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# read the CSV file
df = pd.read_csv(r"C:\Users\10inm\Desktop\spam_filter_bootcamp\emails.csv")
# print the first 5 rows of data
print(df.head())

# print the number of rows and columns
print(df.shape)

# check for duplicates
print(df.duplicated().sum())

# remove duplicates
print(df.drop_duplicates(inplace=True))

# check for missing values
print(df.isna().sum())


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
	# remove punctuation
	no_punctuation = [char for char in text if char not in string.punctuation]
	# store in no_punctuation list
	no_punctuation = ''.join(no_punctuation)
	# tokenization process (split in words)
	clean_words = [word for word in no_punctuation.split() if word.lower() not in
	               stopwords.words('english')]
	return clean_words


# apply the function "process_text" to the column "text"
df['text'].head().apply(process_text)

# convert a collection of text to a matrix of tokens
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

# assign X and y variables
X = messages_bow
y = df['spam']

# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=1)

# get the shape of X (rows, columns)
print(X.shape)

# create and train the Naive Bayes Classifier


# select the model
model = MultinomialNB()

# train the model
model.fit(X_train, y_train)

# check the prediction on the training data
y_pred_train = model.predict(X_train)

# classification report
print("CLASSIFICATION REPORT\n", classification_report(y_train, y_pred_train))

# confusion matrix
print("CONFUSION MATRIX\n", confusion_matrix(y_train, y_pred_train))

# accuracy score
print("ACCURACY SCORE: ", accuracy_score(y_train, y_pred_train))

# use the model to predict on new (test) data
y_pred_test = model.predict(X_test)

# classification report
print("\t\t\tCLASSIFICATION REPORT\n\n", classification_report(y_test, y_pred_test))

# confusion matrix
print("\nCONFUSION MATRIX\n\n", confusion_matrix(y_test, y_pred_test))

# accuracy score
print("\nACCURACY SCORE:", accuracy_score(y_test, y_pred_test))
