
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
