# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:35:28 2021

@author: Sathvik
"""


import pandas as pd
messages = pd.read_csv(r'C:\Users\Dell\Desktop\SMSSpamCollection',sep = '\t',
                       names=["label","message"])

#Data preprocessing
    
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words("english")]
    review = ' '.join(review)
    corpus.append(review)

#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
#you can also select top n no. of features instead of 7098 by using max_features=n inside Countvectorizer
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#classifier model
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,Y_train)

Y_pred = spam_detect_model.predict(X_test)

#accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test, Y_pred)
    
