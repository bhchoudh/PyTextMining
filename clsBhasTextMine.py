import nltk, os, os.path
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from docx import Document
from openpyxl import load_workbook

class clsBhasTextMine:

    def __init__(self):
        self.text = ""


    def myTextPreProcessor(self, text):
        stopWords = set(nltk.corpus.stopwords.words('english'))
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # Get tokenizer with builtin cleaning
        text = tokenizer.tokenize(text)
        for word in text:
            if word in stopWords:
                text.remove(word)  # remove stopwords
            elif not word.isalpha():
                text.remove(word)  # remove special characters
            elif word.isnumeric():
                text.remove(word)
        # print (text)
        #for word, pos in nltk.pos_tag(text):
        #    if (pos == 'NNP' or pos == 'NNPS' or pos == 'PRP' or pos == 'CD'):
        #        text.remove(word)
        print(text)
        # texttoken=[]
        # porter = PorterStemmer()
        # lancaster = LancasterStemmer()
        # for word in text:
        #    texttoken.append(lancaster.stem(word))

        return text

    def myReadDocx(self,fullfilepath):
        doc = Document(fullfilepath)
        data = ''
        for para in doc.paragraphs:
            data = data + ' ' + para.text
        return data