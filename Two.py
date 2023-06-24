import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


# # Load the dataset
# data = pd.read_csv('HateXplain.csv')
# print(data)
# # Split the data into train and test sets
# train, test = train_test_split(data, test_size=0.2, random_state=42)
#
# print(train)
# print(test)
#
# train.to_csv('train.csv', index=False)
# test.to_csv('test.csv', index=False)
# Data ingestion
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df


# Data preprocessing
def preprocess_data(df):
    df = df.fillna("unknown")  # handling missing values
    return df


# Data cleaning
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuations
    text = text.lower()  # convert to lower case
    text = [word for word in word_tokenize(text) if word not in stopwords.words('english')]  # remove stopwords
    return " ".join(text)


# Data exploration
def data_exploration(df):
    print(f'Total records: {len(df)}')
    print(f'Total columns: {len(df.columns)}')
    print(f'Columns: {df.columns}')
    print(f'Null values: {df.isnull().sum()}')


def main():
    # Load data
    train_df, test_df = load_data()

    # Preprocess data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Clean text data
    train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)

    # Data exploration
    data_exploration(train_df)

    # Feature extraction
    tfidf = TfidfVectorizer()
    train_features = tfidf.fit_transform(train_df['comment_text'])

    # Model training
    svc = LinearSVC()
    svc.fit(train_features, train_df['toxic'])

    # Model prediction
    test_features = tfidf.transform(test_df['comment_text'])
    predictions = svc.predict(test_features)

    print(classification_report(test_df['toxic'], predictions))


if __name__ == "__main__":
    main()
