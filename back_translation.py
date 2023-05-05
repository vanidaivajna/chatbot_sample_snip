import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def back_translate(text, tokenizer, model, device='cpu'):
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors='pt').to(device)

    # Generate back translation by passing the input text through the model twice,
    # once to translate to a foreign language and once to translate back to the original language
    translated = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    retranslated = model.generate(tokenizer.encode(translated_text, return_tensors='pt').to(device), 
                                  max_length=128, num_beams=4, early_stopping=True)
    retranslated_text = tokenizer.decode(retranslated[0], skip_special_tokens=True)
    
    return retranslated_text

# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small').to('cpu')

# Test the back_translate function
text = "The quick brown fox jumps over the lazy dog"
augmented_text = back_translate(text, tokenizer, model, 'cpu')
print("Original Text:", text)
print("Augmented Text:", augmented_text)
