import nltk
from nltk.corpus import wordnet

def augment_sentence(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)

    # Find synonyms for each token
    synonyms = []
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
    
    # Replace each token with a random synonym
    augmented_tokens = []
    for token in tokens:
        if token in synonyms:
            augmented_tokens.append(synonyms[random.randint(0, len(synonyms)-1)])
        else:
            augmented_tokens.append(token)
    
    # Join the tokens back into a sentence
    augmented_sentence = ' '.join(augmented_tokens)
    
    return augmented_sentence

  
 #----------------------#0


from textblob import TextBlob

def augment_sentence(sentence):
    # Create a TextBlob object from the input sentence
    blob = TextBlob(sentence)

    # Translate the sentence into a random language
    translated_blob = blob.translate(to=random.choice(['es', 'fr', 'de', 'ja', 'ko', 'zh']))
    
    # Return the translated sentence
    return str(translated_blob)
#----------------------------------------------
from googletrans import Translator

# initialize translator
translator = Translator()

# original text
text = "This is a sample sentence for back-translation."

# translate to Spanish
translation = translator.translate(text, dest='es').text

# translate back to English
back_translation = translator.translate(translation, dest='en').text

print("Original text:", text)
print("Back-translated text:", back_translation)


#--------------------------------------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer

# initialize T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# original text
text = "This is a sample sentence for paraphrasing."

# generate paraphrases
inputs = tokenizer.encode("paraphrase: " + text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
paraphrases = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

print("Original text:", text)
print("Paraphrases:", paraphrases)
#-------------------------------------------------------

import nlpaug.augmenter.word as naw

# original text
text = "This is a sample sentence for synonym replacement."

# initialize augmentation model
aug = naw.SynonymAug(aug_src='wordnet')

# generate augmented text
augmented_text = aug.augment(text)

print("Original text:", text)
print("Augmented text:", augmented_text)

#--------------------------------
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

# initialize the augmentation techniques
aug1 = nas.ContextualWordEmbsForSentenceAug(model_path='bert-base-uncased')
aug2 = nas.AbstSummAug()

# create an augmentation flow
flow = naf.Sequential([aug1, aug2], name='Sequential')

# example input sentence
sentence = 'The quick brown fox jumped over the lazy dog.'

# augment the sentence
augmented_sentences = flow.augment(sentence, n=3)

# print the augmented sentences
print("Original sentence:", sentence)
print("Augmented sentences:")
for augmented_sentence in augmented_sentences:
    print(augmented_sentence)
#----------------------
from transformers import T5ForConditionalGeneration, T5Tokenizer

# initialize T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# original text
text = "This is a sample sentence for paraphrasing."

# generate paraphrases
inputs = tokenizer.encode("paraphrase: " + text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
paraphrases = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

print("Original text:", text)
print("Paraphrases:")
for paraphrase in paraphrases:
    print(paraphrase)
