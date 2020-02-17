import nltk
import pandas as pd
import os, os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document

#nltk.download('punkt')                              #download NLTK packages
#nltk.download('stopwords')                          #download NLTK packages

def myTextPreProcessor(text):
    stopWords = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # Get tokenizer with builtin cleaning
    text = tokenizer.tokenize(text)
    for word in text:
        if word in stopWords:
            text.remove(word)  # remove stopwords
        elif not word.isalpha():
            text.remove(word)  # remove special characters
    return text

#Keep doc, txt files in a folder to read & create vector of words from files
filepath="c:\BhaskarCode\FILES\TextMining\Docs"
os.path.normpath(filepath)                             #Neutralize OS effect of path
corpus =[]                                              #Initiate list to create corpus to pass to vectorizer
for file in os.listdir(filepath):                       #loop through all files in the folder
    fullfilepath=os.path.join(filepath,file)
    if os.path.isfile(fullfilepath):
        #datafile= open(fullfilepath)
        #data = datafile.read().lower()
        doc = Document(fullfilepath)
        data = ''
        for para in doc.paragraphs:
            data = data + ' ' + para.text
        data = myTextPreProcessor(data)
        print(data)
        data = ' '.join(data)                       #converting list back to string to add to a master list
        corpus.append(data)                         #add whole string into master list to be passed to vector
    else:
        continue

#Pass the corpus list as argument to TFIDFVectorizer & store result in data frame & write to excel
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)
#df = pd.DataFrame(vectors.todense(), columns= vectorizer.get_feature_names())
df = pd.DataFrame(vectors.T.todense(), index=vectorizer.get_feature_names())    #T to transpose column & row
df['calc']=df[0]*df[1]
df.sort_values("calc", axis=0,ascending=False,inplace=True,kind="quicksort")
dfwrite=pd.ExcelWriter("c:\BhaskarCode\FILES\TextMining\TFIDF-COSINE\TFIDF.xlsx")
df.to_excel(dfwrite,sheet_name='TFIDF', index=True)
dfwrite.save()
#df=df.head(5)
print(df)
