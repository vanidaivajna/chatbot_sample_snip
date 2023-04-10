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
