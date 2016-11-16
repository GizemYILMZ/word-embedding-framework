import pandas

class MyEvaluation(object):
    def __init__(self):
        self._accuracy = 0.0
        self._AUC = 0.0
        self._macroF1 = 0.0
        self._microF1 = 0.0
        self._F1 = []
        self._Precision = []
        self._Recall = []
        self._Conf_Matrix= pandas.DataFrame

    def _getAccuracy(self):
        return self._accuracy

    def _setAccuracy(self,value):
        self._accuracy=value

    def _getAUC(self):
        return self._AUC

    def _setAUC(self, value):
        self._AUC = value

        return self._Precision

    def _getPrecision(self):
        return self._Precision

    def _setPrecision(self, value):
        self._Precision = value

    def _getRecall(self):
        return self._Recall

    def _setRecall(self, value):
        self._Recall = value

    def _getF1(self):
        return self._F1

    def _setF1(self, value):
        self._F1 = value

    def _getF1_macro(self):
        return self._F1_macro

    def _setF1_macro(self, value):
        self._F1_macro = value

    def _getF1_micro(self):
        return self._F1_micro

    def _setF1_micro(self, value):
        self._F1_micro = value

    def _getConfusionMatrix(self):
        return self._Conf_Matrix

    def _setConfusionMatrix(self, value):
        self._Conf_Matrix = value