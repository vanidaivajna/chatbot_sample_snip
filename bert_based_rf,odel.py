import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel

# Load the data
data = pd.read_csv('intent_data.csv')

# Define the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define a function to generate sentence embeddings using the BERT model
def get_bert_embeddings(text):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_state = outputs[0]
    sentence_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
    return sentence_embedding.detach().numpy()

# Apply the function to the data to get sentence embeddings
data['embedding'] = data['text'].apply(get_bert_embeddings)

# Define the inputs and outputs for training
X = np.stack(data['embedding'].to_numpy())
y = data['intent'].to_numpy()

# Define a function to undersample the majority class
def undersample(X, y, ratio=0.5):
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    minority_X = X[y == minority_class]
    minority_y = y[y == minority_class]
    majority_X = X[y == majority_class]
    majority_y = y[y == majority_class]
    num_samples = int(len(majority_y) * ratio)
    selected_indices = np.random.choice(len(majority_y), num_samples, replace=False)
    downsampled_majority_X = majority_X[selected_indices]
    downsampled_majority_y = majority_y[selected_indices]
    X_new = np.concatenate((minority_X, downsampled_majority_X))
    y_new = np.concatenate((minority_y, downsampled_majority_y))
    return X_new, y_new

# Undersample the majority class
X_new, y_new = undersample(X, y)

# Split the data into training and testing sets
split = 0.8
split_idx = int(split * len(X_new))
train_X, train_y = X_new[:split_idx], y_new[:split_idx]
test_X, test_y = X_new[split_idx:], y_new[split_idx:]

# Train the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_X, train_y)

# Evaluate the classifier on the testing set
pred_y = clf.predict(test_X)
accuracy = accuracy_score(test_y, pred_y)
print('Accuracy:', accuracy)
