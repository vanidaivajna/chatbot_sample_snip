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
