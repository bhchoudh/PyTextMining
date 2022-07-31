import nltk, datetime
import pandas as pd
import os, os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document
from openpyxl import load_workbook
from clsBhasTextMine import clsBhasTextMine

#nltk.download('punkt')                                 #download NLTK packages
#nltk.download('stopwords')                             #download NLTK packages
timestamp =str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+ str(datetime.datetime.now().minute)
txtpreproc = clsBhasTextMine()  #calling common preprocessing function from class
filepath="d:\BhaskarCode\journal\cosine\Docs"           #path to read from files
os.path.normpath(filepath)                              #Neutralize OS effect of path
corpus =[]                                              #Initiate list to create corpus to pass to vectorizer
lengthCheck = []                                        # track how many words are removed

for file in os.listdir(filepath):                       #loop through all files in the folder
    fullfilepath=os.path.join(filepath,file)
    if os.path.isfile(fullfilepath):
        #datafile= open(fullfilepath)
        #data = datafile.read().lower()
        doc = Document(fullfilepath)
        data = ''
        for para in doc.paragraphs:
            data = data + ' ' + para.text

        preLen= len(data.split())
        #data = myTextPreProcessor(data)
        data = txtpreproc.myTextPreProcessor(data)
        postLen =len(data)
        lengthCheck.append([preLen, postLen])
        data = ' '.join(data)                       #converting list back to string to add to a master list
        corpus.append(data)                         #add whole string into master list to be passed to vector
    else:
        continue

dflenchk = pd.DataFrame(lengthCheck,columns=['Pre-TFIDF','Post-TFIDF'])
print(dflenchk)

#Pass the corpus list as argument to TFIDFVectorizer & store result in data frame & write to excel
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)
#df = pd.DataFrame(vectors.todense(), columns= vectorizer.get_feature_names())
df = pd.DataFrame(vectors.T.todense(), index=vectorizer.get_feature_names())    #T to transpose column & row
df['calc']=df[0]*df[1]
df.sort_values("calc", axis=0,ascending=False,inplace=True,kind="quicksort")
print(df)

#output writing in excel with multiple sheet append
excelpath = "d:\BhaskarCode\journal\TFIDF3107.xlsx"
book = load_workbook(excelpath)     #excel needs to exist else it will give error
dfwrite = pd.ExcelWriter(excelpath)
dfwrite.book = book
df.to_excel(dfwrite, sheet_name=timestamp, index=True)
dflenchk.to_excel(dfwrite, sheet_name="lenchk", index=True)
dfwrite.save()
quit()


#####  Dt 31/07/22 ######
#This text preprocessing being a common function is shifted to class clsBhasTextMine
#This function can be removed from this file, kept for past reference only
def myTextPreProcessor(text):
    #stopWords = set(nltk.corpus.stopwords.words('english'))
    stopWords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # Get tokenizer with builtin cleaning
    text = tokenizer.tokenize(text)
    for word in text:
        if word in stopWords:
            text.remove(word)  # remove stopwords
        elif not word.isalpha():
            text.remove(word)  # remove special characters
        elif word.isnumeric():
            text.remove(word)
    #print(text)
    for word, pos in nltk.pos_tag(text):
        if (pos == 'NNP' or pos == 'NNPS' or pos == 'PRP' or pos == 'CD'):
            text.remove(word)
    #print(text)
    # texttoken=[]
    # porter = PorterStemmer()
    # lancaster = LancasterStemmer()
    # for word in text:
    #    texttoken.append(lancaster.stem(word))

    return text

    stopWords = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # Get tokenizer with builtin cleaning
    text = tokenizer.tokenize(text)
    for word in text:
        if word in stopWords:
            text.remove(word)  # remove stopwords
        elif not word.isalpha():
            text.remove(word)  # remove special characters
    return text
