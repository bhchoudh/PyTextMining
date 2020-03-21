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

excelpath = "c:\BhaskarCode\FILES\TextMining\myNaiveBayes.xlsx"
nltk.download('averaged_perceptron_tagger')

def myTextPreProcessor(text):
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
    print (text)
    for word, pos in nltk.pos_tag(text):
        if (pos == 'NNP' or pos == 'NNPS' or pos == 'PRP'):
            text.remove(word)
    print(text)
    # texttoken=[]
    # porter = PorterStemmer()
    # lancaster = LancasterStemmer()
    # for word in text:
    #    texttoken.append(lancaster.stem(word))

    return text

def myReadDocx(fullfilepath):
    doc = Document(fullfilepath)
    data = ''
    for para in doc.paragraphs:
        data = data + ' ' + para.text
    return data

def myModelClassify(filepath,excelpath):      #only to create a excel of classification model by reading DocX
    os.path.normpath(filepath)  # Neutralize OS effect of path
    os.path.normpath(excelpath)
    newdatalist = []
    for file in os.listdir(filepath):  # loop through all files in the folder
        fullfilepath = os.path.join(filepath, file)
        if os.path.isfile(fullfilepath):
            data = myReadDocx(fullfilepath)
            data = myTextPreProcessor(data)
            data = ' '.join(data)  # converting list back to string to add to a master list
            if (len(file) % 2) == 0:
                label = 'High'
            else:
                label = 'Low'
            newdata = [data, label]
            newdatalist.append(newdata)
        else:
            continue

    # print(*newdatalist, sep = "\n")
    dfClassifier = pd.DataFrame(newdatalist, columns=['Text', 'Label'])
    dfwrite = pd.ExcelWriter(excelpath)
    dfClassifier.to_excel(dfwrite, sheet_name='RefData', index=True)
    dfwrite.save()
    return newdatalist

#Used only to create first set of Vector data excel
if not os.path.isfile(excelpath):
    print("No File")
    result = myModelClassify("c:\BhaskarCode\FILES\TextMining\classifyNB", excelpath)

#Read classification tabel from excel with updated label classifiers
dfTrain=pd.read_excel(excelpath)
#column = df.iloc[:, 0]
#column = dfTrain[dfTrain.columns[1]]
#column = column.values.tolist()
text=[]                                 #Argument to Vectorizer Fit Transform
labels=[]                               #Argument to Multinomial Naive Bayes
for index, rows in dfTrain.iterrows():
    text.append(rows['Text'])
    labels.append(rows['Label'])

#Train the model with loaded data using MultinomialNB
train_vectorizer = TfidfVectorizer()
train_vectors = train_vectorizer.fit_transform(text)
model = MultinomialNB().fit(train_vectors, labels)
#train_vectors = train_vectorizer.fit_transform(dfClassifier['Text'])
#model = MultinomialNB().fit(train_vectors, dfClassifier['Label'])
#print(train_vectorizer.get_feature_names())


# Test with new data set, read through docx from a folder & compare with previous vector
filepath = "c:\BhaskarCode\FILES\TextMining\modelNB"
os.path.normpath(filepath)  # Neutralize OS effect of path
predictlist = []
for file in os.listdir(filepath):  # loop through all files in the folder
    fullfilepath = os.path.join(filepath, file)
    if os.path.isfile(fullfilepath):
        data = myReadDocx(fullfilepath)
        data = myTextPreProcessor(data)
        #Naive Bayes Predict takes a list of list, which is a word count of new words in old vector
        testfeature = []
        for word in train_vectorizer.get_feature_names():
            testfeature.append(data.count(word[0]))
        predicted = model.predict([testfeature])
        newdata = [data, predicted]
        predictlist.append(newdata)
    else:
        continue

#print(*predictlist, sep = "\n")
dfTestResult = pd.DataFrame(predictlist, columns=['Text', 'Label'])
book = load_workbook(excelpath)
dfwrite = pd.ExcelWriter (excelpath)
dfwrite.book = book
dfTestResult.to_excel(dfwrite, sheet_name='TestData', index=True)
dfwrite.save()
quit()



##################### Alternate algorithm, not used in this case

X_train, X_test, y_train, y_test = train_test_split(train_vectors, df1['Label'], test_size=0.1, random_state=69)
model = MultinomialNB().fit(X_train, y_train)
predicted = model.predict(X_test)
print(accuracy_score(y_test, predicted))
quit()
NewDoc1 = "this is a test data to be tested against NB claasification model"

TestList = []
def myTestDataPrep(data):
    data = myTextPreProcessor(data)
    data = ' '.join(data)  # converting list back to string to add to a master list
    newdata = [data, '']
    TestList.append(newdata)
    return TestList

TestList = myTestDataPrep(NewDoc1)

# print(*TestList, sep = "\n")
df2 = pd.DataFrame(TestList, columns=['Text', 'Label'])
test_vectors = vectorizer.fit_transform(df2['Text'])
# test_vectors.reshape(1,-1)

predicted = model.predict(test_vectors)
print(predicted)
