import pandas as pd
import re
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
import fasttext
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load the data
data = pd.read_csv('data.csv')

# Step 2: Preprocessing
def preprocess_text(text):
    # Remove symbols and punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove smilies
    text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

data['text'] = data['text'].apply(preprocess_text)

# Step 3: Split data for 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Build FastText model
for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    # Prepare train and validation data for the fold
    train_data = data.iloc[train_idx].copy()
    val_data = data.iloc[val_idx].copy()
    
    # Write train and validation data to file in FastText format
    train_file = f'train_{fold}.txt'
    val_file = f'val_{fold}.txt'
    with open(train_file, 'w') as f:
        for i, row in train_data.iterrows():
            label = '__label__' + str(row['intent'])
            text = row['text']
            f.write(label + ' ' + text + '\n')
    with open(val_file, 'w') as f:
        for i, row in val_data.iterrows():
            label = '__label__' + str(row['intent'])
            text = row['text']
            f.write(label + ' ' + text + '\n')
    
    # Train FastText model on the train data
    model = fasttext.train_supervised(input=train_file, epoch=50)
    
    # Vectorize train and validation data using FastText model
    X_train = np.array([model.get_sentence_vector(text) for text in train_data['text']])
    y_train = train_data['intent']
    X_val = np.array([model.get_sentence_vector(text) for text in val_data['text']])
    y_val = val_data['intent']
    
    # Step 5: Build Random Forest and Naive Bayes models and evaluate on validation set
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    nb = MultinomialNB()
    models = {'Random Forest': rf, 'Naive Bayes': nb}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        print(f'Fold {fold+1} {model_name}: Precision={p:.4f}, Recall={r:.4f}, F1-score={f1:.4f}')
