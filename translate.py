import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the data into a pandas DataFrame
data = pd.read_csv('input.csv')

# Define the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define a function to generate translations
def translate(sentence, source_language, target_language):
    input_str = f'translate {source_language} to {target_language}: {sentence}'
    input_ids = tokenizer.encode(input_str, return_tensors='pt')
    outputs = model.generate(input_ids)
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_sentence

# Apply the translation function to the DataFrame
data['Translated'] = data['Input'].apply(lambda x: translate(x, 'en', 'ar'))

# Save the augmented data to a new CSV file
data.to_csv('translated_data.csv', index=False)
