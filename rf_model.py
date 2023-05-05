import fasttext
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# Load the pre-trained FastText model
model = fasttext.load_model('model.bin')

# Load the labeled dataset
X = []
y = []
with open('data.txt') as f:
    for line in f:
        label, text = line.strip().split('\t')
        X.append(text)
        y.append(label)

# Generate features for the dataset using the FastText model
X_ft = []
for text in X:
    X_ft.append(model.get_sentence_vector(text))

# Initialize the cross-validation splitter
kf = KFold(n_splits=5)

# Initialize the F1 scores list
f1_scores = []

# Perform 5-fold cross-validation
for train_idx, test_idx in kf.split(X_ft):
    # Split the dataset into training and test sets
    X_train = [X_ft[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X_ft[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    # Train a Random Forest classifier on the FastText features
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Make predictions on the test set and calculate the F1 score
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Add the F1 score to the list
    f1_scores.append(f1)

# Print the average F1 score across all folds
print('Average F1 score:', sum(f1_scores) / len(f1_scores))
