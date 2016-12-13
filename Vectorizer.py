import os
import abc
import TaggedLineSentence
from gensim.models import Doc2Vec
import numpy
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import utils

class Vectorizer:
    __metaclass__ = abc.ABCMeta
    def __init__(self,fileNames):
        self.fileNames=fileNames

    def GetfileNames(self):
        return self.test_data
    def SetfileNames(self,fileNames):
        self.fileNames=fileNames
    @abc.abstractmethod
    def get_vectors(self):
        """Returns vectors ."""

class tfidfVectorizer(Vectorizer):
    def __init__(self,fileNames,use_idf):
        self.fileNames=fileNames
        self.use_idf=use_idf
        self.dataset = self.get_vectors()

    @staticmethod
    def preprocess(term):  # term is one file in a one class
        exclude = set(string.punctuation)
        term = ''.join(ch for ch in term if ch not in exclude).lower()
        return term  # term.lower().translate( string.punctuation )

    @staticmethod
    def stemmerTrFps6(term):
        return term[:6]

    @staticmethod
    def stem_tokens(tokens,stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer(item))
        return stemmed

    def tokenize(self,text):
        stemmer = self.stemmerTrFps6
        tokens = nltk.word_tokenize(text)
        stems = self.stem_tokens(tokens, stemmer)
        return stems

    def get_vectors(self):
        tfidf = TfidfVectorizer(min_df=2, preprocessor=self.preprocess,lowercase=False,use_idf=True,tokenizer=self.tokenize)
        docTermMatrix = tfidf.fit_transform((open(f, encoding='utf-8').read() for f in self.fileNames))
        return docTermMatrix


class Doc2vecVectorizer(Vectorizer):

    def __init__(self):
        self.dataset = numpy.array
        self.dataset_labels = numpy.array
        self.file_count = []
        self.total_count=0
        self.labelcount=0

    def get_counts(self,fileNames):
        self.file_count.clear()
        self.total_count=0
        for i in range(len(fileNames)):
            with open(fileNames[i], encoding='utf8') as myfile:
                count = sum(1 for line in myfile)
                self.file_count.append(count)
            self.total_count = self.total_count + count

    # create doc2vec model and train model.
    def get_vectors(self,fileNames,classLabels,model_name,size,min_count,dm,window,iter):
        sources = {'data/train/train-saglik.txt': 'TRAIN_SAGLIK', 'data/train/train-siyaset.txt': 'TRAIN_SIYASI'}
        sources.clear()
        i = 0
        print("Model will be created for files : " , fileNames)
        for file in fileNames:
            sources.__setitem__(file,str(i))
            i += 1

        sentences = TaggedLineSentence.TaggedLineSentence(sources)
        model = Doc2Vec(size=size, min_count=min_count, iter=iter, dm=dm,window=window)
        model.build_vocab(sentences.to_array())
        model.train(sentences)

        model.save(model_name + '.d2v')
        print("Model  ", model_name, "created for files : ", fileNames)

    # create dataset from doc2vec model. Take file names and class labels arrays which exist in doc2vec model
    def create_arrays(self,fileNames,classLabels,input_model,size):
        print("Model : ",input_model)
        print("Filenames :" , fileNames)
        print("Classlabels :", classLabels)
        self.labelcount=len(classLabels)
        self.get_counts(fileNames) # get all files line counts seperately
        print(self.total_count)
        print("Total file count : " ,self.total_count)
        model = Doc2Vec.load(input_model)   # load pre-built model
        train_arrays = numpy.zeros((self.total_count, size))  # create numpy array for classifiers. 100 is because of windows size of doc2vec model
        train_labels = []
        y=0
        for x in range(int(self.labelcount)):
            count = self.file_count[x]
            for i in range(count):
                label = classLabels[x] + '_' + str(i)  # label example is TRAIN-EKONOMI_1
                train_arrays[y] = model.docvecs[label]
                train_labels.append(classLabels[x])
                y += 1
        self.dataset = train_arrays
        self.dataset_labels = train_labels
        print("Train dataset was created")


    def infer_vector(self,fileNames,classLabels,model,size):
        i=0
        x=0
        self.get_counts(fileNames)  # get all files line counts seperately
        print("Model : ",model)
        print("Filenames :" , fileNames)
        print("Classlabels :", classLabels)
        print("Total file count: ",self.total_count)
        test_arrays = numpy.zeros((self.total_count, size))  # create numpy array for classifiers. 100 is because of windows size of doc2vec model
        test_labels = []
        model = Doc2Vec.load(model)  # load pre-built model
        for source in fileNames:
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    sentences = ' '.join(utils.to_unicode(line).split() + [classLabels[i] + '_%s' % item_no])
                    list = sentences.split()
                    inferred_vector = model.infer_vector(list)
                    # print(source)
                    # print(mdl.docvecs.most_similar([inferred_vector]))
                    test_arrays[x] = inferred_vector
                    test_labels.append(classLabels[i])
                    x = x + 1;
            i = i + 1;
        self.dataset = test_arrays
        self.dataset_labels = test_labels
        print("Test vectors were created")

    def find_similar_words(self,model,word):
        model = Doc2Vec.load(model)
        most_similar_words = model.most_similar(word)
        return most_similar_words

    def predict_class(self,model,input_file):
        model = Doc2Vec.load(model)
        with open(input_file, 'r', encoding='utf8') as file:
            data = file.readline()
        split_data= data.split()
        print(split_data)
        inferred_vector = model.infer_vector(split_data)
        classes = model.docvecs.most_similar([inferred_vector],topn=3)
        return classes

