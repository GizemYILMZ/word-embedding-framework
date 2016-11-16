import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import abc
import TaggedLineSentence
from gensim.models import Doc2Vec
import numpy

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
    def __init__(self,fileNames):
        self.fileNames=fileNames
        self.dataset = self.get_vectors()

    @staticmethod
    def preprocess(term):  # term is one file in a one class
        exclude = set(string.punctuation)
        term = ''.join(ch for ch in term if ch not in exclude).lower().replace('\n', ' ')
        return term  # term.lower().translate( string.punctuation )

    def get_vectors(self):
        #tokenizer = tokenize,
        #stop_words='english'
        tfidf = TfidfVectorizer(preprocessor=self.preprocess, lowercase=True)
        docTermMatrix = tfidf.fit_transform((open(f, encoding='utf-8').read() for f in self.fileNames))
        return docTermMatrix


class Doc2vecVectorizer(Vectorizer):

    def __init__(self):
        self.dataset = numpy.array
        self.dataset_labels = numpy.array
        self.dataset_test = numpy.array
        self.dataset_labels_test = numpy.array
        self.file_count = []
        self.total_count=0
        self.labelcount=0

    def get_counts(self,fileNames):
        for i in range(len(fileNames)):
            with open(fileNames[i], encoding='utf8') as myfile:
                count = sum(1 for line in myfile)
                self.file_count.append(count)
            self.total_count = self.total_count + count

    # create doc2vec model and train model.
    def get_vectors(self,fileNames,classLabels):
        sources = {'data/train/train-saglik.txt': 'TRAIN_SAGLIK', 'data/train/train-siyasi.txt': 'TRAIN_SIYASI'}
        sources.clear()
        i = 0
        for file in fileNames:
            sources.__setitem__(file,classLabels[i])
            i += 1

        sentences = TaggedLineSentence.TaggedLineSentence(sources)
        model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
        model.build_vocab(sentences.to_array())
        # create and train model only once then load
        for epoch in range(10):
            model.train(sentences.sentences_perm())
        model.save('./1150haber.d2v')

    # create dataset from doc2vec model. Take file names and class labels arrays which exist in doc2vec model
    def create_arrays(self,fileNames,classLabels):
        self.labelcount=len(classLabels)
        self.get_counts(fileNames) # get all files line counts seperately
        model = Doc2Vec.load('./1150haber.d2v')   # load pre-built model
        test_arrays = numpy.zeros((self.total_count, 100))  # create numpy array for classifiers. 100 is because of windows size of doc2vec model
        test_labels = numpy.zeros(self.total_count)
        keep_x = 0
        y=0
        for x in range(int(self.labelcount)):
            count = self.file_count[x]
            print(count)
            print(classLabels[x])
            for i in range(count):
                label = classLabels[x] + '_' + str(i)  # label example is TRAIN-EKONOMI_1
                print(label)
                test_arrays[y] = model.docvecs[label]
                test_labels[y] = x
                y += 1

            self.dataset = test_arrays
            self.dataset_labels = test_labels

        # train_arrays = numpy.array
        # train_labels = numpy.array
        # y=0
        # x = keep_x+1
        # for z in range(int(self.labelcount/2)):
        #     count = self.file_count[x]
        #     print(count)
        #     print(self.classLabels[x])
        #     for i in range(count):
        #         new = self.classLabels[x] + '_' + str(i)
        #         train_arrays[y] = model.docvecs[new]
        #         train_labels[y] = z
        #         y += 1
        #     x += 1

        # self.dataset_test=test_arrays
        # self.dataset_labels_test=test_labels

