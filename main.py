import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def ingest_data(file_path):
    """Function to load the dataset."""
    data = pd.read_csv(file_path)
    return data


data = ingest_data('your_file_path_here.csv')  # replace with your csv file path




def preprocess_data(data, text_field):
    """Function for data preprocessing and cleaning."""
    # Convert to lowercase
    data[text_field] = data[text_field].str.lower()
    # Remove special characters
    data[text_field] = data[text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return data


data = preprocess_data(data, 'text')  # replace 'text' with your text column name

# Split the data into train and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)


def explore_data(data):
    """Function to perform basic data exploration."""
    # Display the first 5 rows of the dataset
    print(data.head())
    # Display the distribution of classes in the dataset
    print(data['label'].value_counts())  # replace 'label' with your target column name


explore_data(train)




def train_model(X_train, y_train):
    """Function to train the model."""
    # Create a pipeline using TF-IDF and LinearSVC
    model = Pipeline([('tfidf', TfidfVectorizer()),
                      ('clf', LinearSVC())])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Function to evaluate the model."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


X_train = train['text']  # replace 'text' with your text column name
y_train = train['label']  # replace 'label' with your target column name

model = train_model(X_train, y_train)

X_test = test['text']  # replace 'text' with your text column name
y_test = test['label']  # replace 'label' with your target column name

evaluate_model(model, X_test, y_test)


def predict_hate_speech(model, sentence):
    """Function to predict whether a sentence is hate speech."""
    prediction = model.predict([sentence])
    if prediction == 0:  # replace with your non-hate-speech class
        return "This sentence is not hate speech."
    else:
        return "This sentence is hate speech."


print(predict_hate_speech(model, "Fuckin bitch."))
