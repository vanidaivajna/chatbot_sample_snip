import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load BERT module
bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")

# Prepare data
data = pd.read_csv('intent_classification_data.csv')
sentences = data['text'].tolist()
labels = data['intent'].tolist()
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split data into training and testing sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Convert data to BERT input format
def get_bert_input(sentences, bert_module):
    input_tokens = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    input_ids = []
    input_masks = []
    input_segments = []
    for sentence in input_tokens:
        bert_input = bert_module(sentence, signature="tokens", as_dict=True)
        input_ids.append(bert_input["input_word_ids"])
        input_masks.append(bert_input["input_mask"])
        input_segments.append(bert_input["input_type_ids"])
    return input_ids, input_masks, input_segments

train_input_ids, train_input_masks, train_input_segments = get_bert_input(train_sentences, bert_module)
test_input_ids, test_input_masks, test_input_segments = get_bert_input(test_sentences, bert_module)

# Define model
def build_model(max_seq_length):
    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name='input_ids')
    input_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_segments = Input(shape=(max_seq_length,), dtype=tf.int32, name='input_segments')
    pooled_output, sequence_output = bert_module([input_ids, input_mask, input_segments], signature="tokens", as_dict=True)
    output = Dense(units=512, activation='relu')(pooled_output)
    output = Dropout(0.1)(output)
    output = Dense(units=len(le.classes_), activation='softmax')(output)
    model = Model(inputs=[input_ids, input_mask, input_segments], outputs=output)
    return model

# Compile and train model
model = build_model(max_seq_length=128)
model.compile(optimizer=Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([train_input_ids, train_input_masks, train_input_segments], train_labels, epochs=5, batch_size=32)

# Evaluate model
model.evaluate([test_input_ids, test_input_masks, test_input_segments], test_labels)
