import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download("stopwords")

dataset = pd.read_csv("train.csv")
dataset["tweet"][0]
tweet=re.sub("@[\w]", " " , dataset["tweet"][0])



corpus = []
for i in range(0,31962):
    tweet=re.sub("[^a-zA-Z#]" , " " , dataset["tweet"][i])
    tweet=tweet.lower()
    tweet=tweet.split()
    tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words("english"))]
    tweet=" ".join(tweet)
    corpus.append(tweet)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=31615)
x=cv.fit_transform(corpus).toarray()

y=dataset.iloc[:, -1].values