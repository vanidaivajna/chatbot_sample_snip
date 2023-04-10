import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the NLU data into a pandas DataFrame
df_nlu = pd.read_csv('nlu_data.csv')

# Create a TfidfVectorizer object to convert text data into a vectorized format
vectorizer = TfidfVectorizer()

# Convert the sample utterances into a vectorized format
X = vectorizer.fit_transform(df_nlu['sample_utterances'])

# Define the target variable (i.e. the intents)
y = df_nlu['intent']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print a classification report to evaluate the performance of the model
print(classification_report(y_test, y_pred))
