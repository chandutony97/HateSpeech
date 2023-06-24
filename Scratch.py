#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet
from nltk import word_tokenize
import pandas as pd
import random
import numpy as np
# import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt
import pickle
import string
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC, LinearSVC# Support Vector Machine
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import re
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from gensim.models.doc2vec import Doc2Vec


# Preprocessing Functions

# In[3]:


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms
stop = nltk.corpus.stopwords.words('english')

def augment_data(sent):
    words = sent.split()
    words = [w if w not in stop else '@'+w for w in words]
    for i in range(len(words)):
        if not words[i].startswith('@'):
            syn_w = get_synonyms(words[i])
            if syn_w != []:
                w = random.choice(syn_w)
                words[i] = " ".join(w.split('_'))
        else:
            words[i] = words[i][1:]
    return " ".join(words)


def preprocess_text(s):
    s = s.replace('\n',' ')
    s = s.replace('\t',' ')
    s = s.replace(':',' ')
    s = s.replace('#',' ')
    s = s.replace('*','u')
    s = s.replace('@','a')
    s = s.replace('$','s')
    s = s.replace('7','s')
    s = s.replace('2','to')
    s = s.replace('8','ight')
    s = s.replace('&', 'and')
    s = s.translate(str.maketrans('', '', string.punctuation) ) 
    s = s.split()
    s = [i for i in s if i]
    s = [re.sub("[^0-9a-zA-Z]+", "", i) for i in s]
    s = [i for i in s if len(i)>1]    
    return " ".join(s)


def transform_x(df):
    x = df.apply(lambda row : preprocess_text(row['comment_text']), axis=1)
    return pd.DataFrame(x,columns=['comment_text'])

def merge(df1,df2):
    return pd.concat([df1, df2], axis=1)


def drop_faulty_rows(df):
    return df.drop(df[(df['toxic'] == -1.0) & (df['severe_toxic'] == -1.0) & 
                    (df['obscene'] == -1.0) & (df['threat'] == -1.0) & 
                    (df['insult'] == -1.0) & (df['identity_hate'] == -1.0) ].index)
    
def combine_labels(train_df):
    x = np.where(train_df['toxic']+train_df['severe_toxic']+train_df['obscene']
             +train_df['threat']+train_df['insult']+train_df['identity_hate'] > 0, 1, 0)
    return pd.DataFrame(x,columns=['Toxic'])
    


# In[4]:


train_df = pd.read_csv(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\train.csv\train.csv")


# In[5]:


train_df.head(5)
     


# In[6]:


X = transform_x(train_df)
X.head()


# In[7]:


Y = combine_labels(train_df)
Y.head()


# Test Data

# In[8]:


test_df = pd.read_csv(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\test.csv\test.csv")
y_test = pd.read_csv(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\test_labels.csv\test_labels.csv")
y_test.head(3)
     


# In[9]:


x_test = transform_x(test_df)
df_col_merged = merge(x_test,y_test)
df_col_merged.head()


# In[10]:


test_df = drop_faulty_rows(df_col_merged)


# In[11]:


x_test = test_df['comment_text']
y_test = combine_labels(test_df)


# Make my own Embeddings

# In[12]:


import os
import re
import time

from gensim.models import Word2Vec
from tqdm import tqdm

tqdm.pandas()


# In[13]:


X.comment_text


# In[14]:


sentences = pd.concat([X.comment_text,x_test],axis=0)
train_sent = list(sentences.progress_apply(str.split).values)


# In[15]:


start_time = time.time()

model = Word2Vec(sentences=train_sent, 
                 sg=1, 
                 vector_size=100,  
                 workers=4)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')
model.wv.save_word2vec_format(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\custom_glove_100d.txt")


# In[16]:


start_time = time.time()

model = Word2Vec(sentences=train_sent, 
                 sg=1, 
                 vector_size=300,  
                 workers=4)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')
model.wv.save_word2vec_format(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\custom_glove_300d.txt")





start_time = time.time()

model = Word2Vec(sentences=train_sent, 
                 sg=1, 
                 vector_size=768,  
                 workers=4)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')
model.wv.save_word2vec_format(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\custom_glove_768d.txt")





from gensim.models.doc2vec import Doc2Vec, TaggedDocument
train_sent = list(X.comment_text.progress_apply(str.split).values)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_sent)]
model = Doc2Vec(documents, vector_size=300, window=8, min_count=5, workers=4, dm = 1, epochs=20)



from gensim.test.utils import get_tmpfile
fname = get_tmpfile(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\doc2vec_model")
model.save(fname)


# With custom embeddings

# In[20]:


from gensim.models import KeyedVectors
from collections import defaultdict

w2v = KeyedVectors.load_word2vec_format(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\custom_glove_768d.txt")
n_dim = 768


# In[21]:


tf_idf = TfidfVectorizer()
tf_idf.fit(X['comment_text'])
max_idf = max(tf_idf.idf_)
tf_idf_dict = defaultdict(
            lambda: max_idf,
            [(w, tf_idf.idf_[i]) for w, i in tf_idf.vocabulary_.items()])
     


# In[22]:


def get_word_vec(word):
    try:
         return w2v.word_vec(word)
    except:
        return np.zeros(n_dim) 
vect_get_word_vec = np.vectorize(get_word_vec)

def get_sentence_embed(sent):
    words = np.array(sent.split())
    if len(words)==0:
        return np.zeros(n_dim)
    word_vecs = np.array([vect_get_word_vec(x) for x in words])
    return np.average(word_vecs,axis=0)

def get_sentence_embed_tf_idf(sent):
    global tf_idf_dict
    words = np.array(sent.split())
    if len(words) == 0:
        return np.zeros(n_dim)
    word_vecs = np.array([vect_get_word_vec(x) for x in words])
    for i in range(len(words)):
        word_vecs[i] = tf_idf_dict[words[i]]*word_vecs[i]
    return np.average(word_vecs,axis=0)


# In[23]:


X_train_sent = X.comment_text.to_numpy()
sent_embed_X_train = np.stack([get_sentence_embed(x) for x in X_train_sent])
sent_embed_tfidf_X_train = np.stack([get_sentence_embed_tf_idf(x)  for x in X_train_sent])


# In[24]:


sent_embed_X_train.shape


# In[25]:


x_test_sent = x_test.to_numpy()
sent_embed_X_test = np.stack([get_sentence_embed(x) for x in x_test_sent])
sent_embed_tfidf_X_test = np.stack([get_sentence_embed_tf_idf(x)  for x in x_test_sent])


# Linear svm
# 
# Normal Avg

# In[26]:


lsvm  = LinearSVC()
lsvm.fit(sent_embed_X_train, Y.Toxic.to_numpy().ravel())


# In[27]:


pickle.dump(lsvm,open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\lsvm_emb_on_Train_avged.pkl",'wb'))


# In[28]:


lsvm_avg_emb = pickle.load(open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\lsvm_emb_on_Train_avged.pkl",'rb'))


# Analysis of embedding size

# In[29]:


y_pred = lsvm.predict(sent_embed_X_test)
print(classification_report(y_test, y_pred)) #100d


# In[30]:


y_pred = lsvm.predict(sent_embed_X_test)
print(classification_report(y_test, y_pred)) #300d


# In[31]:


y_pred = lsvm_avg_emb.predict(sent_embed_X_test)
print(classification_report(y_test, y_pred)) #768d
     


# TFIDF Weighted Avg

# In[32]:


lsvm_tfidf_emb = LinearSVC(max_iter=100)
lsvm_tfidf_emb.fit(sent_embed_tfidf_X_train, Y.Toxic.to_numpy().ravel())
pickle.dump(lsvm_tfidf_emb, open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\lsvm_emb_on_Train_tfidf_avged",'wb'))


# In[33]:


lsvm_tfidf_emb = pickle.load(open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\lsvm_emb_on_Train_tfidf_avged",'rb'))


# In[34]:


lsvm_tfidf_emb
     


# In[35]:


y_pred = lsvm_tfidf_emb.predict(sent_embed_X_test)
print(classification_report(y_test, y_pred)) 


# SVM RBF
# 
# Normal Avg

# In[36]:


emb_svc = SVC(kernel='rbf')
emb_svc.fit(sent_embed_X_train, Y.Toxic.to_numpy().ravel())
pickle.dump(emb_svc, open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\svm_rbf_emb_on_Train_avged.pkl",'wb'))


# In[37]:


svm_rbf_avg_emb = pickle.load(open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\svm_rbf_emb_on_Train_avged.pkl",'rb'))
     


# In[39]:


y_pred = svm_rbf_avg_emb.predict(sent_embed_X_test)
print(classification_report(y_test, y_pred)) #768d


# TFIDF Weighted Avg

# In[40]:


emb_svc = SVC(kernel='rbf')
emb_svc.fit(sent_embed_X_train, Y.Toxic.to_numpy().ravel())
pickle.dump(emb_svc, open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\svm_rbf_emb_on_Train_tfidf_avged.pkl",'wb'))


# In[ ]:


svm_rbf_tfidf_emb = pickle.load(open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\svm_rbf_emb_on_Train_tfidf_avged.pkl",'rb'))


# Doc2Vec

# In[ ]:




doc2vec = Doc2Vec.load(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\doc2vec_model")



X_train_sent = X.comment_text.to_numpy()
sent_embed_X_train = np.stack([doc2vec.infer_vector(word_tokenize(x)) for x in X_train_sent])




sent_embed_X_train.shape



x_test_sent = x_test.to_numpy()
sent_embed_X_test = np.stack([doc2vec.infer_vector(word_tokenize(x)) for x in x_test_sent])


# SVM RBF



emb_svc = SVC(kernel='rbf')
emb_svc.fit(sent_embed_X_train, Y.Toxic.to_numpy().ravel())
pickle.dump(emb_svc, open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\svm_rbf_d2v_emb.pkl",'wb'))



svm_rbf_d2v_emb = pickle.load(open(r"C:\Users\rittw\Documents\Humber\NLP\NLP_Humber\jigsaw-toxic-comment-classification-challenge\svm_rbf_d2v_emb.pkl",'rb'))



y_pred = svm_rbf_d2v_emb.predict(sent_embed_X_test)
print(classification_report(y_test, y_pred)) 

