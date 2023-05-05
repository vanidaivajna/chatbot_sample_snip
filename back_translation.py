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



from transformers import AutoModelWithLMHead, AutoTokenizer

# Load the T5 model and tokenizer
model = AutoModelWithLMHead.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Set the source and target languages for backtranslation
source_lang = "en"
target_lang = "fr"

# Define the input sentence to be backtranslated
input_sentence = "This is a test sentence to be backtranslated."

# Preprocess the input sentence and convert it to a tensor
input_ids = tokenizer.encode(input_sentence, return_tensors="pt")

# Generate the backtranslated sentence using the T5 model
translated = model.generate(input_ids=input_ids, 
                             max_length=128, 
                             num_beams=4, 
                             early_stopping=True, 
                             no_repeat_ngram_size=2, 
                             do_sample=True, 
                             top_k=50, 
                             top_p=0.95, 
                             temperature=0.7, 
                             num_return_sequences=1, 
                             decoder_start_token_id=model.config.decoder_start_token_id, 
                             eos_token_id=model.config.eos_token_id, 
                             pad_token_id=model.config.pad_token_id, 
                             bos_token_id=model.config.bos_token_id, 
                             use_cache=True)

# Decode the backtranslated sentence and remove any special tokens
translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)

# Print the original input sentence and the backtranslated sentence
print("Input sentence: ", input_sentence)
print("Backtranslated sentence: ", translated_sentence)
