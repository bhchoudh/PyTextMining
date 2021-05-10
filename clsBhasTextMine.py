import nltk, os, os.path
from docx import Document

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