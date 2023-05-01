import numpy as np
import pandas as pd
import json
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, Callback

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

# Define a callback function to print the training progress for each epoch
class PrintEpochProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch+1}/{num_epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['categorical_accuracy']:.4f}")

# Train and evaluate the model on each fold
fold_accuracies = []
fold_f1_scores = []
fold_precisions = []
fold_recalls = []

for i, (train_indices, val_indices) in enumerate(folds):
    print(f"Training fold {i+1}")
    df_train = df_patterns.iloc[train_indices].reset_index(drop=True)
    df_val = df_patterns.iloc[val_indices].reset_index(drop=True)

    train_input_ids = np.zeros((len(df_train), seq_len))
    train_attention_masks = np.zeros((len(df_train), seq_len))

    for j, pattern in enumerate(df_train['text']):
        input_ids, attention_mask = tokenize_inputs(pattern)
        train_input_ids[j, :] = input_ids
        train_attention_masks[j, :] = attention_mask

    val_input_ids = np.zeros((len(df_val), seq_len))
    val_attention_masks = np.zeros((len(df_val), seq_len))

    for j, pattern in enumerate(df_val['text']):
        input_ids, attention_mask = tokenize_inputs(pattern)
        val_input_ids[j, :] = input_ids
        val_attention_masks[j, :] = attention_mask

    train_labels = tf.keras.utils.to_categorical(df_train['intent'].factorize()[0])
    val_labels = tf.keras.utils.to_categorical(df_val['intent'].factorize()[0])

    model = create_model()

    # Train the model for 5 epochs with early stopping
    num_epochs = 5
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        history = model.fit([train_input_ids, train_attention_masks], train_labels,
                            validation_data=([val_input_ids, val_attention_masks], val_labels),
                            epochs=1, batch_size=16, callbacks=[es])
        print("Training loss: {:.3f}, Training accuracy: {:.3f}".format(history.history['loss'][0], 
                                                                         history.history['categorical_accuracy'][0]))
        print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(history.history['val_loss'][0], 
                                                                             history.history['val_categorical_accuracy'][0]))

    # Evaluate the model on the validation set
    predictions = np.argmax(model.predict([val_input_ids, val_attention_masks]), axis=-1)
    targets = np.argmax(val_labels, axis=-1)
    fold_accuracies.append(np.mean(predictions == targets))
    report = classification_report(targets, predictions, output_dict=True)
    fold_f1_scores.append(report['macro avg']['f1-score'])
    fold_precisions.append(report['macro avg']['precision'])
    fold_recalls.append(report['macro avg']['recall'])

# Print the average performance metrics over all folds
print("Average accuracy: {:.3f}".format(np.mean(fold_accuracies)))
print("Average F1-score: {:.3f}".format(np.mean(fold_f1_scores)))
print("Average precision: {:.3f}".format(np.mean(fold_precisions)))
print("Average recall: {:.3f}".format(np.mean(fold_recalls)))


