import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the data into a pandas DataFrame
data = pd.read_csv('input.csv')

# Define the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define a function to generate Arabic translations of English sentences
def translate_to_arabic(sentence):
    input_str = 'translate English to Arabic: ' + sentence
    input_ids = tokenizer.encode(input_str, return_tensors='pt')
    outputs = model.generate(input_ids)
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_sentence

# Define a function to back-translate Arabic sentences to English
def back_translate_to_english(sentence):
    input_str = 'translate Arabic to English: ' + sentence
    input_ids = tokenizer.encode(input_str, return_tensors='pt')
    outputs = model.generate(input_ids)
    back_translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return back_translated_sentence

# Apply the translation and back-translation functions to the DataFrame
data['Arabic Translation'] = data['English'].apply(translate_to_arabic)
data['Back Translation'] = data['Arabic Translation'].apply(back_translate_to_english)

# Save the augmented data to a new CSV file
data.to_csv('augmented_data.csv', index=False)
