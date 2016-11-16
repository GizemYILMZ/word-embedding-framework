from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from Evaluation import MyEvaluation


class Estimation(object):
    def __init__(self):
        return

    def evaluate_testOnTrainingSet(self,dataset,classlabels,classifier):
        predictions = classifier.predict(dataset)
        accuracy = metrics.accuracy_score(classlabels, predictions)
        precision = precision_score(classlabels, predictions, average=None)
        recall = recall_score(classlabels, predictions, average=None)

        F1 = metrics.f1_score(classlabels, predictions, average=None)
        F1_macro = metrics.f1_score(classlabels, predictions, average='macro')
        F1_micro = metrics.f1_score(classlabels, predictions, average='micro')

        cm = confusion_matrix(classlabels, predictions)

        myEval = MyEvaluation()
        myEval._setAccuracy(accuracy.mean())
        myEval._setPrecision(precision)
        myEval._setRecall(recall)
        myEval._setF1(F1)
        myEval._setF1_macro(F1_macro)
        myEval._setF1_micro(F1_micro)
        myEval._setConfusionMatrix(cm)

        return myEval



    def evaluate_nFoldCV(self,dataset,classlabels,classifier,n):
        # k_fold = KFold(len(y_test), n_folds=n, shuffle=True, random_state=0)
        predictions=cross_validation.cross_val_predict(classifier,dataset,classlabels,cv=n)
        # score=cross_val_score(classifier, dataset, classlabels,cv=n,scoring='accuracy')  avg is same as metrics.accuracy_score

        accuracy = metrics.accuracy_score(classlabels, predictions)
        precision = precision_score(classlabels, predictions, average=None)
        recall = recall_score(classlabels, predictions, average=None)

        F1=metrics.f1_score(classlabels, predictions, average=None)
        F1_macro=metrics.f1_score(classlabels, predictions, average='macro')
        F1_micro=metrics.f1_score(classlabels, predictions, average='micro')

        cm = confusion_matrix(classlabels, predictions)

        myEval = MyEvaluation()
        myEval._setAccuracy(accuracy.mean())
        myEval._setPrecision(precision)
        myEval._setRecall(recall)
        myEval._setF1(F1)
        myEval._setF1_macro(F1_macro)
        myEval._setF1_micro(F1_micro)
        myEval._setConfusionMatrix(cm)
        return myEval



    def evaluate_trainTestSplit(self,dataset,classLabels,classifier,testPercantage):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset, classLabels, test_size=testPercantage,random_state=0)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = classifier.score(X_test, y_test)
        precision = precision_score(y_test, predictions, average=None)
        recall = recall_score(y_test, predictions, average=None)

        F1=metrics.f1_score(y_test, predictions, average=None)
        F1_macro=metrics.f1_score(y_test, predictions, average='macro')
        F1_micro=metrics.f1_score(y_test, predictions, average='micro')
        # auc = roc_auc_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        myEval = MyEvaluation()
        myEval._setAccuracy(accuracy)
        myEval._setPrecision(precision)
        myEval._setRecall(recall)
        myEval._setF1(F1)
        myEval._setF1_macro(F1_macro)
        myEval._setF1_micro(F1_micro)
        myEval._setConfusionMatrix(cm)
        # myEval._setAUC(auc)

        return myEval


