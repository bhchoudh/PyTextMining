import nltk
import pandas as pd
import os, os.path, time
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt')                              #download NLTK packages
#nltk.download('stopwords')                          #download NLTK packages
#Keep doc, txt files in a folder to read & create vector of words from files
filepath="c:\BhaskarCode\FILES\TextMining"              #input('Parent directory:- ')
os.path.normpath(filepath)                             #Neutralize OS effect of path
corpus =[]                                              #Initiate list to create corpus to pass to vectorizer
for file in os.listdir(filepath):                       #loop through all files in the folder
    fullfilepath=os.path.join(filepath,file)
    if os.path.isfile(fullfilepath):
        datafile= open(fullfilepath)
        data = datafile.read().lower()
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')       #Get tokenizer with uiltin cleaning
        data = tokenizer.tokenize(data)                         #Tokenizing +basic cleaning into list
        #manually remove stopwords & special characters
        stopWords = set(nltk.corpus.stopwords.words('english'))
        for word in data:
            if word in stopWords:
                data.remove(word)                   #remove stopwords
            elif not word.isalpha():
                data.remove(word)                   #remove special characters
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
dfwrite=pd.ExcelWriter("c:\BhaskarCode\FILES\TFIDF.xlsx")
df.to_excel(dfwrite,sheet_name='TFIDF', index=True)
dfwrite.save()
print(df)
