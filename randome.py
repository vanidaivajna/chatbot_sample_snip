import pandas as pd
import numpy as np

# load your pandas dataframe with 5 categories
df = pd.read_csv('your_data.csv')

# define a list of categories with more than 300 rows
selected_categories = []
for category in df['category'].unique():
    if df[df['category']==category].shape[0] > 300:
        selected_categories.append(category)

# create an empty dataframe to store the selected rows
selected_df = pd.DataFrame()

# randomly select 300 rows for each category with more than 300 rows
for category in selected_categories:
    selected_rows = df[df['category']==category].sample(n=300, random_state=42)
    selected_df = pd.concat([selected_df, selected_rows])

# keep the rest of the categories as it is in the original dataframe
rest_df = df[~df['category'].isin(selected_categories)]

# concatenate the selected and rest dataframes
final_df = pd.concat([selected_df, rest_df])
