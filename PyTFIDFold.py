import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt')
#nltk.download('stopwords')

filepath1= "c:\BhaskarCode\FILES\TextMining\TextMining1.txt"
filepath2= "c:\BhaskarCode\FILES\TextMining\TextMining2.txt"
file1 = open(filepath1)
file2 = open(filepath2)
data1=file1.read().lower()
data2=file2.read().lower()

#Data Cleaning (Stop word & punkutuation removal & tokenize)
#Option 1 by calling NLTK library
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
data1 = tokenizer.tokenize(data1)
data2 = tokenizer.tokenize(data2)

#Option 2 : perform manual stop word & punctuation removal
stopWords = set(nltk.corpus.stopwords.words('english'))
#data1 = nltk.word_tokenize(data1)
for word in data1:
    if word in stopWords:
        data1.remove(word)
    elif not word.isalpha():
        data1.remove(word)
#print(data1)
#data2 = nltk.word_tokenize(data2)
for word in data2:
    if word in stopWords:
        data2.remove(word)
    elif not word.isalpha():
        data2.remove(word)
#print(data2)

#TFIDF vectorize, first convert list to string
data1=' '.join(data1)
data2=' '.join(data2)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([data1, data2])
#df = pd.DataFrame(vectors.todense(), columns= vectorizer.get_feature_names())
df = pd.DataFrame(vectors.T.todense(), index=vectorizer.get_feature_names())    #T to transpose column & row
print(df)

dfwrite=pd.ExcelWriter("c:/BhaskarCode/FILES/TFIDF.xlsx")
df.to_excel(dfwrite,sheet_name='TFIDF', index=True)
dfwrite.save()
