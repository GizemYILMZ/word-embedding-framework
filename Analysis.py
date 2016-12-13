import os
import Vectorizer
import string
from sklearn import svm
from Estimation import Estimation
import time
import configparser
from sklearn.linear_model import LogisticRegression

class Analysis(object):
    def __init__(self):
        self.vectorizer=""
        self.classifier=""
        self.fileNames=[]
        self.classLabels=[]
        self.config = configparser.ConfigParser()

    def get_vectorizer(self):
        return self.vectorizer
    def get_classifier(self):
        return self.classifier
    def set_vectorizer(self,value):
        self.vectorizer=value
    def set_classifier(self,value):
        self.classifier=value

# create doc2vec input takes a path and label as an argument and puts together all file to one file for every class
    def create_Doc2vec_input(self,path,label):
        print("Working on creating Doc2vec input in path %s and label is %s" %(path,label))
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
        print("Creating Doc2vec input is finished")

    # find class labels and file names in a path  for tfidf
    def readcorpus_TfIdf(self,path):
        print("Reading corpus names from path", path)
        self.fileNames.clear()
        self.classLabels.clear()
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = subdir + os.path.sep + file
                self.fileNames.append(file_path)
                self.classLabels.append(subdir[len(path):])

    # find class labels and file names in a path  for Doc2vec
    def readcorpus_Doc2Vec(self,path):
        print("Reading corpus names from path", path)
        self.fileNames.clear()
        self.classLabels.clear()
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = subdir + os.path.sep + file
                label = file[0:len(file) - 4].upper()  # one label  example is "train-magazin"
                self.classLabels.append(label)
                self.fileNames.append(file_path)
        print("Filenames : ", self.fileNames)
        print("Classlabels : ",self.classLabels)

    def create_doc2vec_model(self,path,model_name,size,min_count,dm,window,iter):
        d2v_object = Analysis()
        d2v_object.readcorpus_Doc2Vec(path)  # read corpus for D2V. Necesarry  for all runs
        Doc2Vec_vectorizer = Vectorizer.Doc2vecVectorizer()  # create doc2vec vectorizer object
        Doc2Vec_vectorizer.get_vectors(d2v_object.fileNames, d2v_object.classLabels,model_name,size,min_count,dm,window,iter)  # create models. This is necessasry only once
        return Doc2Vec_vectorizer;

    def test_doc2vec(self,train_path_d2v, test_path_d2v, input_model, percentage,n, size,output_file_name):
        Doc2Vec_vectorizer_train = Vectorizer.Doc2vecVectorizer()
        self.readcorpus_Doc2Vec(train_path_d2v)  # read corpus for D2V. Necesarry  for all runs
        Doc2Vec_vectorizer_train.infer_vector(self.fileNames, self.classLabels,input_model,size)  # create train data array from model
        train_arrays = Doc2Vec_vectorizer_train.dataset
        train_labels = Doc2Vec_vectorizer_train.dataset_labels

        Doc2Vec_vectorizer_test = Vectorizer.Doc2vecVectorizer()  # create doc2vec vectorizer object
        self.readcorpus_Doc2Vec(test_path_d2v)  # read corpus for D2V. Necesarry  for all runs
        Doc2Vec_vectorizer_test.infer_vector(self.fileNames, self.classLabels,input_model,size)  # create train data array from model
        test_array = Doc2Vec_vectorizer_test.dataset
        test_labels = Doc2Vec_vectorizer_test.dataset_labels

        print("-----------------------------TEST DOC2VEC-------------------------------------")
        estimation=Estimation()
        start = time.time()
        clf = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)


        myEval_trainTestSplit_SVM=estimation.evaluate_trainTestSplit(train_arrays, train_labels, clf, percentage)
        end=time.time()
        print("\n -------------------split 60%-40% with SVM / DOC2VEC------------------------ \n")
        print("Accuracy :", myEval_trainTestSplit_SVM._getAccuracy())
        print("Precision:", myEval_trainTestSplit_SVM._getPrecision())
        print("Recall   :", myEval_trainTestSplit_SVM._getRecall())
        print("F1       :" , myEval_trainTestSplit_SVM._getF1())
        print("F1 macro :" , myEval_trainTestSplit_SVM._getF1_macro())
        print("F1 micro :" , myEval_trainTestSplit_SVM._getF1_micro())
        print("Confusion Matrix :\n " , myEval_trainTestSplit_SVM._getConfusionMatrix())
        print("Time : ", end - start)

        self.config.add_section('test-doc2vec-split 60-40-SVM')
        time_ = end - start
        technique = 'split 60-40'
        classifier_name = 'SVM'
        self.write_to_config_file(myEval_trainTestSplit_SVM, 'test-doc2vec-split 60-40-SVM', time_, technique,classifier_name)

        start = time.time()
        classifier = clf.fit(train_arrays,train_labels)
        myEval_nFoldCV_SVM=estimation.evaluate_nFoldCV(test_array,test_labels,classifier,n)
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

        time_ = end-start
        technique = 'KFoldCV'
        classifier_name='SVM'
        self.config.add_section('test-doc2vec-KFoldCV-SVM')
        self.write_to_config_file(myEval_nFoldCV_SVM, 'test-doc2vec-KFoldCV-SVM', time_, technique,classifier_name)


        clf_LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                        intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
        start = time.time()
        classifier =  clf_LR.fit(train_arrays, train_labels)
        myEval_nFoldCV_LR = estimation.evaluate_nFoldCV(test_array, test_labels, classifier, n)
        end = time.time()

        print("\n -----------------", n, "fold CV with LR / DOC2VEC------------------------ \n")
        print("Accuracy :", myEval_nFoldCV_LR._getAccuracy())
        print("Precision:", myEval_nFoldCV_LR._getPrecision())
        print("Recall   :", myEval_nFoldCV_LR._getRecall())
        print("F1       :", myEval_nFoldCV_LR._getF1())
        print("F1 macro :", myEval_nFoldCV_LR._getF1_macro())
        print("F1 micro :", myEval_nFoldCV_LR._getF1_micro())
        print("Confusion Matrix :\n ", myEval_nFoldCV_LR._getConfusionMatrix())
        print("Time : ", end - start)

        time_ = end - start
        technique = 'KFoldCV'
        classifier_name = 'Logistic Regression'
        self.config.add_section('test-doc2vec-KFoldCV-LR')
        self.write_to_config_file(myEval_nFoldCV_LR, 'test-doc2vec-KFoldCV-LR', time_, technique,classifier_name)

        with open(output_file_name, 'a+') as configfile:
            self.config.write(configfile)

    def test_tfidf(self,train_path_tfidf,test_path_tfidf,percentage,n,output_file_name,use_idf):

        self.readcorpus_TfIdf(train_path_tfidf)
        Tfidf_Vectorizer_train = Vectorizer.tfidfVectorizer(self.fileNames,use_idf)
        classLabels = self.classLabels
        dataset = Tfidf_Vectorizer_train.dataset
        estimation_tfidf = Estimation()


        print("-------------------TEST TF-IDF----------------------")

        # start = time.time()
        clf_tfidf = svm.SVC(kernel='linear', C=1.0, decision_function_shape=None)
        # myEval_trainTestSplit_SVM = estimation_tfidf.evaluate_trainTestSplit(dataset, classLabels, clf_tfidf,percentage)
        # end=time.time()
        # print("\n -------------------split 60%-40% with SVM / TF-IDF------------------------ \n")
        # print("Accuracy :", myEval_trainTestSplit_SVM._getAccuracy())
        # print("Precision:", myEval_trainTestSplit_SVM._getPrecision())
        # print("Recall   :", myEval_trainTestSplit_SVM._getRecall())
        # print("F1       :", myEval_trainTestSplit_SVM._getF1())
        # print("F1 macro :", myEval_trainTestSplit_SVM._getF1_macro())
        # print("F1 micro :", myEval_trainTestSplit_SVM._getF1_micro())
        # print("Confusion Matrix :\n ", myEval_trainTestSplit_SVM._getConfusionMatrix())
        # print("Time : ", end - start)
        #
        # self.config.add_section('test-tf/idf-split 60-40')
        # time_ = end - start
        # technique = 'split 60-40'
        # classifier_name = 'SVM'
        # self.write_to_config_file(myEval_trainTestSplit_SVM, 'test-tf/idf-split 60-40', time_, technique,classifier_name)

        start = time.time()
        classifier = clf_tfidf.fit(dataset, classLabels)
        break_time = time.time()

        self.readcorpus_TfIdf(test_path_tfidf)
        Tfidf_Vectorizer_test = Vectorizer.tfidfVectorizer(self.fileNames,use_idf)
        classLabels_test = self.classLabels
        dataset_test = Tfidf_Vectorizer_test.dataset
        start_2 = time.time()
        myEval_nFoldCV_SVM = estimation_tfidf.evaluate_nFoldCV(dataset_test, classLabels_test, classifier, n)
        end = time.time()
        time_ = (end - start_2) + (break_time - start)

        print("\n -----------------", n, "fold CV with SVM / TF-IDF------------------------ \n")
        print("Accuracy :", myEval_nFoldCV_SVM._getAccuracy())
        print("Precision:", myEval_nFoldCV_SVM._getPrecision())
        print("Recall   :", myEval_nFoldCV_SVM._getRecall())
        print("F1       :", myEval_nFoldCV_SVM._getF1())
        print("F1 macro :", myEval_nFoldCV_SVM._getF1_macro())
        print("F1 micro :", myEval_nFoldCV_SVM._getF1_micro())
        print("Confusion Matrix :\n ", myEval_nFoldCV_SVM._getConfusionMatrix())
        print("Time : ", time_ )

        classifier_name = 'SVM'
        technique='KfoldCV'
        self.config.add_section('test-tf/idf-KfoldCV-SVM')
        self.write_to_config_file(myEval_nFoldCV_SVM, 'test-tf/idf-KfoldCV-SVM', time_, technique,classifier_name)

        self.readcorpus_TfIdf(train_path_tfidf)
        Tfidf_Vectorizer_train = Vectorizer.tfidfVectorizer(self.fileNames, use_idf)
        classLabels = self.classLabels
        dataset = Tfidf_Vectorizer_train.dataset

        clf_LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                    intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
        start = time.time()
        classifier = clf_LR.fit(dataset, classLabels)
        self.readcorpus_TfIdf(test_path_tfidf)
        classLabels_test = self.classLabels
        dataset_test = Tfidf_Vectorizer_test.dataset
        myEval_nFoldCV_LR = estimation_tfidf.evaluate_nFoldCV(dataset_test, classLabels_test, classifier, n)
        end = time.time()

        print("\n -----------------", n, "fold CV with LR / TF-IDF------------------------ \n")
        print("Accuracy :", myEval_nFoldCV_LR._getAccuracy())
        print("Precision:", myEval_nFoldCV_LR._getPrecision())
        print("Recall   :", myEval_nFoldCV_LR._getRecall())
        print("F1       :", myEval_nFoldCV_LR._getF1())
        print("F1 macro :", myEval_nFoldCV_LR._getF1_macro())
        print("F1 micro :", myEval_nFoldCV_LR._getF1_micro())
        print("Confusion Matrix :\n ", myEval_nFoldCV_LR._getConfusionMatrix())
        print("Time : ", end - start)

        time_ = end - start
        technique='KfoldCV'
        classifier_name='Logistic Regression'
        self.config.add_section('test-tf/idf-KfoldCV-LR')
        self.write_to_config_file(myEval_nFoldCV_LR, 'test-tf/idf-KfoldCV-LR', time_, technique,classifier_name)
        with open(output_file_name, 'a+') as configfile:
            self.config.write(configfile)

    def write_to_config_file(self,evalution,test_name,time,technique,classifier):
        self.config.set(test_name, 'classifier', classifier)
        self.config.set(test_name, 'technique', technique)
        self.config.set(test_name, 'accuracy',str(evalution._getAccuracy()))
        self.config.set(test_name, 'precision', str(evalution._getPrecision()))
        self.config.set(test_name, 'recall', str(evalution._getRecall()))
        self.config.set(test_name, 'f1', str(evalution._getF1()))
        self.config.set(test_name, 'f1 macro', str(evalution._getF1_macro()))
        self.config.set(test_name, 'f1 micro', str(evalution._getF1_micro()))
        self.config.set(test_name, 'confusion matrix', str(evalution._getConfusionMatrix()))
        self.config.set(test_name, 'time', str(time))


