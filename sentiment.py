from textblob import TextBlob
import pandas as pd

# load the conversation data into a pandas dataframe
conversation_data = pd.read_csv("conversation_data.csv")

# define a function to get the sentiment polarity of a text
def get_sentiment(text):
    """
    Returns the sentiment polarity of a text using TextBlob library
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

# group the conversation data by context id
grouped_data = conversation_data.groupby('context_id')

# create a new dataframe to store the sentiment analysis results
sentiment_data = pd.DataFrame(columns=['context_id', 'sentiment'])

# loop through each group and perform sentiment analysis on the human text
for name, group in grouped_data:
    human_text = group[group['speaker']=='human']['text'].str.cat(sep='. ')
    sentiment_score = get_sentiment(human_text)
    sentiment_data = sentiment_data.append({'context_id': name, 'sentiment': sentiment_score}, ignore_index=True)

# print the sentiment analysis results
print(sentiment_data)
