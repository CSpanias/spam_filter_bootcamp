import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix



# Importing dataset and separating labels
train_data = pd.read_csv("D:\ML\Datasets\emails.csv")
y_train = train_data['spam']
train_data.drop(labels='spam', axis=1, inplace=True )

# Creating bag of words
vectorizer = CountVectorizer(max_features=100)
X = vectorizer.fit_transform(train_data.text)
features = vectorizer.get_feature_names()
X_train = X.toarray()

# Splitting the dataset
state = 12
train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=train_size, stratify=y_train, random_state=state)

# Building trees
gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0, verbose=3) # can specify loss function
gb_clf.fit(X_train, y_train)
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
predictions = gb_clf.predict(X_test)

print('Confusion Matrix:') # printing confusion matrix using test values of Y and the predictive value of y
print(confusion_matrix(y_test, predictions))

# printing confusion matrix in the colored format seen below in output
cm = confusion_matrix(y_val, predictions)
cm
class_names=[0, 1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu",fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# printing classification report
print("Classification Report:")
print(classification_report(y_val, predictions))