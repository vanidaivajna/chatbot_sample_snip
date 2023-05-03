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

