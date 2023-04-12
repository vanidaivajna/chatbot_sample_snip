import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import words

# Load the data into a Pandas DataFrame
data = pd.read_csv('chatbot_data.csv')

# Get the list of known words from the nltk.corpus.words corpus
word_list = set(words.words())

# Define a function to check for typos in a sentence
def check_typos(sentence):
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)
    
    # Get a list of words that are not in the word list
    misspelled = [word for word in tokens if word.lower() not in word_list]
    
    # If there are any misspelled words, return them as a string, otherwise return None
    if len(misspelled) > 0:
        return (True, ", ".join(misspelled))
    else:
        return (False, None)

# Apply the function to the 'human_text' column of the DataFrame and create two new columns with the results
data[['contains_typos', 'typo_words']] = data['human_text'].apply(check_typos).apply(pd.Series)

# Convert the 'contains_typos' column to 'Yes' or 'No' instead of True or False
data['contains_typos'] = data['contains_typos'].map({True: 'Yes', False: 'No'})

# Print the first 10 rows of the DataFrame to check the results
print(data.head(10))














import pandas as pd
import nltk

# Load the data into a Pandas DataFrame
data = pd.read_csv('data.csv')

# Define a function to check the grammar of a sentence using nltk.parse
def check_grammar(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    
    # Parse the sentence and create a DependencyGraph
    try:
        parser = nltk.parse.corenlp.CoreNLPParser()
        parse = next(parser.raw_parse(sentence))
        graph = parse.to_dependency_graph()
        return True
    except:
        return False

# Apply the function to the 'text' column of the DataFrame and create a new column with the results
data['is_grammatically_correct'] = data['text'].apply(check_grammar)

















import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load conversation data
conv_data = pd.read_csv('conversation_data.csv')

# Load NLU data
nlu_data = pd.read_csv('nlu_data.csv')

# Combine the two datasets into one
data = pd.concat([conv_data['human_text'], nlu_data['Sample Utterances']])

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data)

# Find the optimal number of clusters using silhouette score
scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    scores.append(silhouette_score(X, kmeans.labels_))
    
optimal_k = scores.index(max(scores)) + 2

# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(X)

# Print the clusters and their corresponding utterances
clusters = {}
for i, label in enumerate(kmeans.labels_):
    if label not in clusters:
        clusters[label] = [data[i]]
    else:
        clusters[label].append(data[i])
        
for label, utterances in clusters.items():
    print(f"Cluster {label}: {utterances}")

    
    
import pandas as pd
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from wordcloud import WordCloud

# Load text data as a pandas dataframe
df = pd.DataFrame({
    "Text": [
        "The quick brown fox jumps over the lazy dog.",
        "The lazy dog, peeved to be labeled lazy, jumped quickly over the sleeping cat.",
        "The quick brown fox runs beside the slow hedgehog."
    ]
})

# Tokenize the text in the dataframe
df["Tokens"] = df["Text"].apply(nltk.word_tokenize)

# Generate bigrams for each row in the dataframe
bigram_measures = BigramAssocMeasures()
df["Bigrams"] = df["Tokens"].apply(lambda x: BigramCollocationFinder.from_words(x).nbest(bigram_measures.raw_freq, 10))

# Flatten the list of bigrams
bigrams = [item for sublist in df["Bigrams"].tolist() for item in sublist]

# Generate the word cloud from the list of bigrams
wordcloud = WordCloud(width=800, height=400, max_words=50, background_color="white").generate_from_frequencies(nltk.FreqDist(bigrams))

import pandas as pd
from wordcloud import WordCloud

# Load the data into a Pandas DataFrame
data = pd.read_csv('data.csv')

# Extract the text column from the DataFrame as a list
text = data['text'].tolist()

# Tokenize the text and generate bigrams
tokens = [word.lower() for sent in text for word in sent.split()]
bigrams = list(nltk.bigrams(tokens))
bigram_strings = ["_".join(bigram) for bigram in bigrams]

# Generate the word cloud from the list of bigram strings
wordcloud = WordCloud(width=800, height=400, max_words=50, background_color="white").generate(" ".join(bigram_strings))
