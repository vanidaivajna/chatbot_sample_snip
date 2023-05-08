import pandas as pd
import re
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data into pandas DataFrame
data = pd.read_csv('data.csv')

# Define function to preprocess text
def preprocess(text):
    # Remove symbols, numbers, and smilies
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r':\)', '', text)
    text = re.sub(r':\(', '', text)
    return text

# Preprocess text data
data['text'] = data['text'].apply(preprocess)

# Split data into input (text) and output (intent name) variables
X = data['text']
y = data['intent']

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store evaluation results
rf_precision = []
rf_recall = []
rf_f1 = []
nb_precision = []
nb_recall = []
nb_f1 = []

# Loop through k-fold splits
for train_index, test_index in kf.split(X):
    # Split data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    
    # Fit and transform training data to TF-IDF vectors
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Apply SVD to reduce dimensionality of TF-IDF vectors
    svd = TruncatedSVD(n_components=100)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    
    # Initialize models
    rf = RandomForestClassifier()
    nb = MultinomialNB()
    
    # Train models on transformed training data
    rf.fit(X_train_svd, y_train)
    nb.fit(X_train_svd, y_train)
    
    # Transform test data to TF-IDF vectors and apply SVD
    X_test_tfidf = tfidf.transform(X_test)
    X_test_svd = svd.transform(X_test_tfidf)
    
    # Make predictions using trained models
    rf_pred = rf.predict(X_test_svd)
    nb_pred = nb.predict(X_test_svd)
    
    # Evaluate model performance using precision, recall, and F1 score
    rf_precision.append(precision_score(y_test, rf_pred, average='weighted'))
    rf_recall.append(recall_score(y_test, rf_pred, average='weighted'))
    rf_f1.append(f1_score(y_test, rf_pred, average='weighted'))
    nb_precision.append(precision_score(y_test, nb_pred, average='weighted'))
    nb_recall.append(recall_score(y_test, nb_pred, average='weighted'))
    nb_f1.append(f1_score(y_test, nb_pred, average='weighted'))

# Print evaluation results as table
print('Random Forest Results')
print('---------------------')
print('Precision\tRecall\t\tF1 Score')
for i in range(5):
    print(f'{rf_precision[i]:.4f}\t\t{rf_recall[i]:.4f}\t\t{rf_f1[i]:.4f}')
print(f'Mean\t\t{sum(rf_precision)/5:.4f}\t\t{sum

# Print the results as a table
print(f"{'RandomForestClassifier':<30} {'MultinomialNB':<30}")
print(f"{'Precision':<15} {'Recall':<15} {'F1 Score':<15} {'Precision':<15} {'Recall':<15}
