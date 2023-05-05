import pandas as pd

# Load the labeled data into a pandas DataFrame
data = pd.read_csv('labeled_data.csv')


from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

with open('train.txt', 'w') as f:
    for i, row in train_data.iterrows():
        label = '__label__' + str(row['label'])
        text = row['text']
        f.write(label + ' ' + text + '\n')
import fasttext

# Train a supervised FastText model on the train.txt file
model = fasttext.train_supervised('train.txt')
# Evaluate the model on the validation set
result = model.test('val.txt')
print(result.precision, result.recall, result.f1score)
with open('val.txt', 'w') as f:
    for i, row in val_data.iterrows():
        label = '__label__' + str(row['label'])
        text = row['text']
        f.write(label + ' ' + text + '\n')
# Save the trained model to a binary file
model.save_model('model.bin')
# Load the saved model from the binary file
model = fasttext.load_model('model.bin')

# Make predictions on new data
text = 'This is a new example to classify.'
pred = model.predict(text)
print(pred)

