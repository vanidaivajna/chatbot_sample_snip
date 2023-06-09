import pandas as pd

# Assuming you have a dataframe called 'df' with columns 'id', 'A', and 'B'
# and missing values in column B are represented as NaN

# Group the dataframe by 'id'
grouped_df = df.groupby('id')

# Define a function to fill missing values in column B with any present value in the same ID group
def fill_with_present_value(group):
    # Get a list of non-missing values in column B for the group
    present_values = group['B'].dropna().tolist()
    # If there are any non-missing values, fill missing values with one of them
    if present_values:
        group['B'].fillna(present_values[0], inplace=True)
    return group

# Apply the fill_with_present_value function to each group in the dataframe
df_filled = grouped_df.apply(fill_with_present_value)


#-----------------------
def count_words(text):
    words = text.split()
    return len(words)

# Apply the function to the 'text' column
df['word_count'] = df['text'].apply(count_words)


def count_words(text):
    words = text.split()
    return len(words)

# Apply the function to the 'text' column
df['word_count'] = df['text'].apply(count_words)



import pandas as pd

# Create sample dataframe
data = {'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': ['red', 'green', 'blue'],
        'd': ['small', 'medium', 'large'],
        'e': ['circle', 'square', 'triangle']}

df = pd.DataFrame(data)

# Melt columns c, d, and e into a single column called Category
df_melt = pd.melt(df, id_vars=['a', 'b'], value_vars=['c', 'd', 'e'], var_name='Category')

# Drop the original columns c, d, and e
df_melt = df_melt.drop(['variable'], axis=1)

print(df_melt)











import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder

# Load text data as a pandas dataframe
df = pd.DataFrame({
    "Text": [
        "The quick brown fox jumps over the lazy dog.",
        "The lazy dog, peeved to be labeled lazy, jumped quickly over the sleeping cat.",
        "The quick brown fox runs beside the slow hedgehog."
    ]
})

# Tokenize the text in the dataframe
df["Tokens"] = df["Text"].apply(word_tokenize)

# Generate bigrams for each row in the dataframe
bigram_measures = BigramAssocMeasures()
df["Bigrams"] = df["Tokens"].apply(lambda x: BigramCollocationFinder.from_words(x).nbest(bigram_measures.raw_freq, 10))

# Generate trigrams for each row in the dataframe
trigram_measures = TrigramAssocMeasures()
df["Trigrams"] = df["Tokens"].apply(lambda x: TrigramCollocationFinder.from_words(x).nbest(trigram_measures.raw_freq, 10))

# Print the resulting dataframe
print(df)












import pandas as pd
import nltk
from nltk import word_tokenize
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
df["Tokens"] = df["Text"].apply(word_tokenize)

# Generate bigrams for each row in the dataframe
bigram_measures = BigramAssocMeasures()
df["Bigrams"] = df["Tokens"].apply(lambda x: BigramCollocationFinder.from_words(x).nbest(bigram_measures.raw_freq, 10))

# Flatten the list of bigrams
bigrams = [item for sublist in df["Bigrams"].tolist() for item in sublist]

# Generate the word cloud from the list of bigrams
wordcloud = WordCloud(width=800, height=400, max_words=50, background_color="white").generate_from_frequencies(nltk.FreqDist(bigrams))

# Display the word cloud
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
