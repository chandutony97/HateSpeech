# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import json

# Load JSON data into a pandas DataFrame
with open('dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv('HateXplain.csv', index=False)

# Load the dataset
dataset = pd.read_csv('HateXplain.csv')

# Extract the features and the target variable
features = dataset['text']
target = dataset['label']

# Partition the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a CountVectorizer instance to transform the text data into token count matrix
count_vectorizer = CountVectorizer()
features_train_transformed = count_vectorizer.fit_transform(features_train)
features_test_transformed = count_vectorizer.transform(features_test)

# Use Multinomial Naive Bayes model for training
nb_model = MultinomialNB()
nb_model.fit(features_train_transformed, target_train)

# Use the model to make predictions
target_pred = nb_model.predict(features_test_transformed)

# Display the classification report
print(classification_report(target_test, target_pred))
