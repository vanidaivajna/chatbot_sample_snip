import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import fasttext

# Load the labeled data into a pandas DataFrame
data = pd.read_csv('labeled_data.csv')

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Write the training and validation data to files in the FastText format
with open('train.txt', 'w') as f:
    for i, row in train_data.iterrows():
        label = '__label__' + str(row['label'])
        text = row['text']
        f.write(label + ' ' + text + '\n')

with open('val.txt', 'w') as f:
    for i, row in val_data.iterrows():
        label = '__label__' + str(row['label'])
        text = row['text']
        f.write(label + ' ' + text + '\n')

# Train a supervised FastText model on the train.txt file
model = fasttext.train_supervised('train.txt')

# Extract embeddings for each text in the training and validation sets
train_embeddings = []
for i, row in train_data.iterrows():
    text = row['text']
    embedding = model.get_sentence_vector(text)
    train_embeddings.append(embedding)

val_embeddings = []
for i, row in val_data.iterrows():
    text = row['text']
    embedding = model.get_sentence_vector(text)
    val_embeddings.append(embedding)

# Convert embeddings to numpy arrays
X_train = np.array(train_embeddings)
X_val = np.array(val_embeddings)

# Extract labels
y_train = train_data['label']
y_val = val_data['label']

# Train random forest classifier on the embeddings
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the random forest classifier on the validation set
y_pred = clf.predict(X_val)
accuracy = np.mean(y_pred == y_val)
print(f'Accuracy: {accuracy:.3f}')
