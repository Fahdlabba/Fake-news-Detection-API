import re
import nltk
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pk
nltk.download('stopwords')
with open('model/model.pkl','rb') as f:
  model=pk.load(f)

classes=['Fake','Real']


def stemming(content):
  port_stem=PorterStemmer()
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english') ]
  stemmed_content=' '.join(stemmed_content)
  return stemmed_content


def prediction(content):
  content=stemming(content)
  prediction=model.predict([content])
  return classes[prediction[0]]


