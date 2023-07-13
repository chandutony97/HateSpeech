import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load the data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# Combining all toxic labels
def combine_labels(df):
    df['Toxic'] = df[df.columns[2:]].sum(axis=1)
    df['Toxic'] = df['Toxic'] >= 1
    df['Toxic'] = df['Toxic'].astype(int)
    return df


# Combine toxic labels in train and test datasets
train_df = combine_labels(train_df)
test_df = combine_labels(test_df)

# Splitting train dataset into X and Y
X = train_df.loc[:, ['comment_text']]
Y = train_df.loc[:, ['Toxic']]

# Splitting test dataset into x_test and y_test
x_test = test_df.loc[:, ['comment_text']]
y_test = combine_labels(test_df)

# Further splitting training data for validation
X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Visualize the counts of each class
# sns.countplot(x='Toxic', data=Y)
# plt.title('Counts of each class')
# plt.show()
#
# # Checking the length of comments
# comment_len = train_df.comment_text.str.split().apply(len)
# plt.hist(comment_len, bins=30)
# plt.title('Comment length in words')
# plt.show()

# Create a pipeline with TF-IDF Vectorizer and LinearSVC
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC()),
])

# Fit the model with training data
model.fit(X['comment_text'], Y['Toxic'])

# Evaluate on validation data
predictions = model.predict(X_val['comment_text'])
print(classification_report(Y_val['Toxic'], predictions))


def is_toxic(sentence):
    prediction = model.predict([sentence])
    if prediction[0] == 0:
        return False
    else:
        return True


# def predict_hate_speech(sentence):
#     prediction = model.predict([sentence])
#     if prediction[0] == 0:
#         return "This statement is predicted as non-toxic."
#     else:
#         return "This statement is predicted as toxic."
#
#
# # Test the function
# print(predict_hate_speech("I love this!"))
# print(predict_hate_speech("You are so stupid."))

def predict_hate_speech():
    while True:
        try:
            sentence = input("Please enter a sentence or 'exit' to quit: ")
            #  user enters exit if they want to exit the loop
            if sentence.lower() == '[exit]':
                break
            if not isinstance(sentence, str):
                raise ValueError("Input must be a string.")
            prediction = model.predict([sentence])
            if prediction[0] == 0:
                print("This statement is predicted as non-toxic.")
            else:
                print("This statement is predicted as toxic.")
        except ValueError as e:
            print("An error occurred:", e)
        except Exception as e:
            print("An unexpected error occurred:", e)


# Test the function
predict_hate_speech()
