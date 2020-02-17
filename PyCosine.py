import nltk
import xlrd
import pandas as pd
import os, os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

#nltk.download('punkt')                              #download NLTK packages
#nltk.download('stopwords')                          #download NLTK packages
#Keep doc, txt files in a folder to read & create vector of words from files
filepath="c:\BhaskarCode\FILES\TextMining\Docs"
#df=pd.read_excel("c:\BhaskarCode\FILES\TFIDF.xlsx")

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

ranking=[]
for file in os.listdir(filepath):                       #loop through all files in the folder
    fullfilepath=os.path.join(filepath,file)
    if os.path.isfile(fullfilepath):
        #os.path.normpath(fullfilepath)

        # Reading Excel file
        #df=pd.read_excel(fullfilepath)
        #df = df.head(20)
        #column = df.iloc[:, 0]
        #column = df[df.columns[0]]
        #column = column.values.tolist()

        # Defining reference dataset to compare Cosine similarity
        refdata = "10 + years experience, with 2+ years in Telecom industry Excellent communication & presentation skill & customer facing experience Containers – Dockers, Kubernetes & Helm Software Development Python CICD/ DevOps / Automation - Jenkins , Ansible, Puppet  etc Consulting, presales & E2E architecting experience Opensource forum upstream contribution Ability to lead multi location virtual teams in matrix set up Excellent communication skill & customer facing experience Deployment experience in OpenStack integration with Juniper Contrail SDN Installing, configuring and troubleshooting various OpenStack services using FUEL GUI and CLI Linux system administration experience Containers Dockers, Kubernetes & Helm Openstack Deployment & system integration experience"
        refdata = myTextPreProcessor(refdata)

        #Reading DOCx file, creating string by appending paragraphs & vectorizing
        doc = Document(fullfilepath)
        column =''
        for para in doc.paragraphs:
            column= column+' ' +para.text
        column= myTextPreProcessor(column)

        refdata = ' '.join(refdata)
        column = ' '.join(column)
        corpus = [refdata, column]

        print("*****************************************" )
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)
        df = pd.DataFrame(vectors.todense(), columns= vectorizer.get_feature_names())
        cosine= cosine_similarity(vectors)
        print(df)
        print(cosine)
        cosine = cosine [0][1]
        ranking.append([file, cosine])
    else:
        continue

dfrank=pd.DataFrame(ranking,columns=['Name', 'Rating'])
dfrank.sort_values("Rating", axis=0,ascending=False,inplace=True,kind="quicksort")
print(dfrank)
dfwrite = pd.ExcelWriter("c:\BhaskarCode\FILES\RankCosine.xlsx")
dfrank.to_excel(dfwrite, sheet_name="Cosine", index=True)
dfwrite.save()
