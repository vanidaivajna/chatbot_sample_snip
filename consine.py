import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a sample DataFrame with a column of sentences
df = pd.DataFrame({'sentences': ['The quick brown fox jumps over the lazy dog.',
                                 'The quick brown fox jumps over the lazy dog and runs away.',
                                 'A brown dog and a black dog are playing together.',
                                 'My sister is fond of chocolate cake.',
                                 'I enjoy reading books in my free time.']})

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute the TF-IDF matrix for the sentences
tfidf_matrix = vectorizer.fit_transform(df['sentences'])

# Compute the cosine similarity matrix between the sentences
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Print the cosine similarity matrix
print(cosine_sim_matrix)
