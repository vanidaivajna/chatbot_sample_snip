import numpy as np
import pandas as pd
import json
import random
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

from transformers import BertTokenizer, TFAutoModel

# Load the dataset
with open('Intent.json') as f:
    data = json.load(f)

# Create a pandas dataframe with pattern and intent
df_patterns = pd.DataFrame(data['intents'])[['text', 'intent']]

# Balance the dataset by oversampling minority classes
def balance_data(df):
    df_intent = df['intent']
    max_counts = df_intent.value_counts().max() #max number of examples for a class
    
    new_df = df.copy()
    for i in df_intent.unique():
        i_count = int(df_intent[df_intent == i].value_counts())
        if i_count < max_counts:
            i_samples = df[df_intent == i].sample(max_counts - i_count, replace = True, ignore_index = True)
            new_df = pd.concat([new_df, i_samples])
    return new_df

df_patterns = balance_data(df_patterns)

# Split the data into 5 folds
num_samples = len(df_patterns)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kfold.split(df_patterns))

# Tokenize the input patterns and create attention masks
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

seq_len = 256

def tokenize_inputs(text):
    tokens = tokenizer.encode_plus(text,
                                   max_length=seq_len,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

# Define the model
def create_model():
    input_ids = tf.keras.layers.Input(shape=(seq_len,), dtype='int32')
    attention_masks = tf.keras.layers.Input(shape=(seq_len,), dtype='int32')
    bert_model = TFAutoModel.from_pretrained('bert-base-cased')
    bert_outputs = bert_model(input_ids, attention_mask=attention_masks)[1]
    outputs = tf.keras.layers.Dense(len(df_patterns['intent'].unique()), activation='softmax')(bert_outputs)
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=[outputs])
    model.compile(optimizer=Adam(lr=2e-5), loss='categorical_crossentropy', metrics=[CategoricalAccuracy()])
    return model

# Train and evaluate the model on each fold
fold_accuracies = []
for i, (train_indices, val_indices) in enumerate(folds):
    print(f"Training fold {i+1}")
    df_train = df_patterns.iloc[train_indices].reset_index(drop=True)
    df_val = df_patterns.iloc[val_indices].reset_index(drop=True)

    train_input_ids = np.zeros((len(df_train), seq_len))
    train_attention_masks = np.zeros
