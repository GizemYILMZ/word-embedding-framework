import os
import Vectorizer
import string
from sklearn import svm
from Estimation import Estimation
import time

class Analysis(object):
    def __init__(self):
        self.train_data=""
        self.test_data=""
        self.vectorizer=""
        self.classifier=""
        self.fileNames=[]
        self.classLabels=[]
    def get_train_data(self):
        return self.train_data
    def get_test_data(self):
        return self.test_data
    def get_vectorizer(self):
        return self.vectorizer
    def get_classifier(self):
        return self.classifier
    def set_train_data(self,value):
        self.train_data=value
    def set_test_data(self,value):
        self.test_data=value
    def set_vectorizer(self,value):
        self.vectorizer=value
    def set_classifier(self,value):
        self.classifier=value

# create doc2vec input takes a path and label as an argument and puts together all file to one file for every class
    def create_Doc2vec_input(self,path,label):
        exclude = set(string.punctuation)
        new_path=" "
        token_list=[]
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = subdir + os.path.sep + file
                token_list.append(file_path)
                new_path = "data/" + label + "-" + subdir[len(path):] + ".txt"   #target path
            for f in token_list:
                with open(f, 'r', encoding='utf8') as myfile:
                    data = myfile.read()
                    data = ''.join(ch for ch in data if ch not in exclude).lower().replace('\n', '')  # convert all uppercase characters to lowercase and remove punct.
                    target = open(new_path, "a+", encoding='utf8')  # write to target path
                target.write(data + '\n')
            token_list.clear()

    # find class labels and file names in a path  for tfidf
    def readcorpus_TfIdf(self,path):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = subdir + os.path.sep + file
                self.fileNames.append(file_path)
                self.classLabels.append(subdir[len(path):])

    # find class labels and file names in a path  for Doc2vec
    def readcorpus_Doc2Vec(self,path):
        self.fileNames.clear()
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = subdir + os.path.sep + file
                label = file[0:len(file) - 4].upper()  # one label  example is "train-magazin"
                self.classLabels.append(label)
                self.fileNames.append(file_path)

#DOC2VEC PATHS
path_1150haber_d2v='data/1150haber_d2v/'
path_1150haber_d2v_test='data/1150haber_d2v/test/'
path_1150haber_d2v_train='data/1150haber_d2v/train/'
path_news_d2v='data/news_d2v/'
path_news_d2v_train='data/news_d2v/train'
path_news_d2v_test='data/news_d2v/test'

#TFIDF PATHS
path_1150haber_tfidf='data/1150haber_tfidf/'
path_news_tfidf='data/news_tfidf/'


Doc2Vec_object=Analysis()  # create Analysis object for doc2vec
# Doc2Vec_object.create_Doc2vec_input(path,"test")   # this line only necesarry for first run

Doc2Vec_object.readcorpus_Doc2Vec(path_1150haber_d2v_train)   # read corpus for D2V. Necesarry  for all runs
# print(Doc2Vec_object.fileNames)
# print(Doc2Vec_object.classLabels)

Doc2Vec_vectorizer=Vectorizer.Doc2vecVectorizer()  # create doc2vec vectorizer object
# Doc2Vec_vectorizer.get_vectors(Doc2Vec_object.fileNames,Doc2Vec_object.classLabels)  # create models. This is necessasry only once
Doc2Vec_vectorizer.create_arrays(Doc2Vec_object.fileNames,Doc2Vec_object.classLabels)   # create train data array from model
dataset=Doc2Vec_vectorizer.dataset
classLabels=Doc2Vec_vectorizer.dataset_labels

# ----------TEST---------
estimation=Estimation()
percentage=0.4

start=time.time()
clf = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)
end=time.time()

myEval_trainTestSplit_SVM=estimation.evaluate_trainTestSplit(dataset,classLabels,clf,percentage)
print("\n -------------------split 60%-40% with SVM / DOC2VEC------------------------ \n")
print("Accuracy :", myEval_trainTestSplit_SVM._getAccuracy())
print("Precision:", myEval_trainTestSplit_SVM._getPrecision())
print("Recall   :", myEval_trainTestSplit_SVM._getRecall())
print("F1       :" , myEval_trainTestSplit_SVM._getF1())
print("F1 macro :" , myEval_trainTestSplit_SVM._getF1_macro())
print("F1 micro :" , myEval_trainTestSplit_SVM._getF1_micro())
print("Confusion Matrix :\n " , myEval_trainTestSplit_SVM._getConfusionMatrix())
print("Time : ", end - start)


n = 10
start=time.time()
classifier = clf.fit(dataset , classLabels)
myEval_nFoldCV_SVM=estimation.evaluate_nFoldCV(dataset,classLabels,classifier,n)
end=time.time()

print("\n -----------------" , n ,"fold CV with SVM / DOC2VEC------------------------ \n")
print("Accuracy :", myEval_nFoldCV_SVM._getAccuracy())
print("Precision:", myEval_nFoldCV_SVM._getPrecision())
print("Recall   :", myEval_nFoldCV_SVM._getRecall())
print("F1       :" , myEval_nFoldCV_SVM._getF1())
print("F1 macro :" , myEval_nFoldCV_SVM._getF1_macro())
print("F1 micro :" , myEval_nFoldCV_SVM._getF1_micro())
print("Confusion Matrix :\n " , myEval_nFoldCV_SVM._getConfusionMatrix())
print("Time : ", end - start)


Tfidf_object=Analysis()
Tfidf_object.readcorpus_TfIdf(path_1150haber_tfidf)
classLabels=Tfidf_object.classLabels
Tfidf_Vectorizer=Vectorizer.tfidfVectorizer(Tfidf_object.fileNames)
dataset=Tfidf_Vectorizer.dataset
# print(classLabels)
# print(dataset)

estimation_tfidf=Estimation()

percentage=0.4
start=time.time()
clf_tfidf = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)
myEval_trainTestSplit_SVM=estimation_tfidf.evaluate_trainTestSplit(dataset,classLabels,clf_tfidf,percentage)
end=time.time()

print("\n -------------------split 60%-40% with SVM / TF-IDF------------------------ \n" )
print("Accuracy :", myEval_trainTestSplit_SVM._getAccuracy())
print("Precision:", myEval_trainTestSplit_SVM._getPrecision())
print("Recall   :", myEval_trainTestSplit_SVM._getRecall())
print("F1       :" , myEval_trainTestSplit_SVM._getF1())
print("F1 macro :" , myEval_trainTestSplit_SVM._getF1_macro())
print("F1 micro :" , myEval_trainTestSplit_SVM._getF1_micro())
print("Confusion Matrix :\n " , myEval_trainTestSplit_SVM._getConfusionMatrix())
print("Time : ", end - start)


n=10
start=time.time()
classifier = clf_tfidf.fit(dataset , classLabels)
myEval_nFoldCV_SVM=estimation_tfidf.evaluate_nFoldCV(dataset,classLabels,classifier,n)
end=time.time()

print("\n -----------------" , n ,"fold CV with SVM / TF-IDF------------------------ \n")
print("Accuracy :", myEval_nFoldCV_SVM._getAccuracy())
print("Precision:", myEval_nFoldCV_SVM._getPrecision())
print("Recall   :", myEval_nFoldCV_SVM._getRecall())
print("F1       :" , myEval_nFoldCV_SVM._getF1())
print("F1 macro :" , myEval_nFoldCV_SVM._getF1_macro())
print("F1 micro :" , myEval_nFoldCV_SVM._getF1_micro())
print("Confusion Matrix :\n " , myEval_nFoldCV_SVM._getConfusionMatrix())
print("Time : ", end - start)
