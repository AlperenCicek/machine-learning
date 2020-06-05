import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QComboBox, QFileDialog, QLineEdit, QMainWindow
import sys
from PyQt5.QtGui import QFont
from prompt_toolkit.widgets import Dialog

class Product:
    def __init__(self):
        self.xTrain = None
        self.yTrain = None
        self.xTrue = None
        self.yTrue = None
        self.test = None
        self.dataSet = None
        self.model = None
        self.MSE = None
        self.RMSE = None
        self.intercept = None
        self.coef = None
        self.formula = ""

    def trainModel(self, train_X_Value, train_Y_Value):
        self.dataSet = self.dataSet
        self.xTrain = self.dataSet[[train_X_Value]]
        self.yTrain = self.dataSet[[train_Y_Value]]
        self.model = SVR(kernel = 'linear', C = 1.0).fit(self.dataSet[[train_X_Value]], self.dataSet[[train_Y_Value]])
        self.intercept = self.model.intercept_
        self.coef = self.model.coef_
        self.formula = str(round(self.model.intercept_[0], 2)) + " + {0}".format(train_X_Value[0:2]) + " * " + str(round(self.model.coef_[0][0], 2))
        g = sns.regplot(self.xTrain, self.yTrain, ci = None, label = train_X_Value, scatter_kws = {'color' : 'b', 's' : 9})
        g.set_ylabel(train_Y_Value)
        g.set_xlabel(train_X_Value)
        plt.legend()
        plt.show()

    def testModel(self, true_x_value, true_y_value):
        self.xTrue = self.test[[true_x_value]]
        self.yTrue = self.test[[true_y_value]]
        pro.MSE = mean_squared_error(self.yTrue, pro.model.predict(self.xTrue))
        pro.RMSE = np.sqrt(mean_squared_error(self.yTrue, pro.model.predict(self.xTrue)))

class GUI(QMainWindow):
    def __init__(self, x, y, width, height, title):
        super(GUI, self).__init__()
        self.inp = ""
        self.trainCSVName = ""
        self.testCSVName = ""
        self.setGeometry(x, y, width, height)
        self.setWindowTitle(title)
        self.initUI()

    def initUI(self):
        self.labelMSE = QtWidgets.QLabel(self)
        self.labelMSE.setText("MSE = ")
        self.labelMSE.move(220, 200)

        self.valueMSE = QtWidgets.QLabel(self)
        self.valueMSE.setText("")
        self.valueMSE.move(260, 200)

        self.labelRMSE = QtWidgets.QLabel(self)
        self.labelRMSE.setText("RMSE = ")
        self.labelRMSE.move(220, 220)

        self.valueRMSE = QtWidgets.QLabel(self)
        self.valueRMSE.setText("")
        self.valueRMSE.move(260, 220)

        self.labelCoef = QtWidgets.QLabel(self)
        self.labelCoef.setText("Coef = ")
        self.labelCoef.move(20, 200)

        self.valueCoef = QtWidgets.QLabel(self)
        self.valueCoef.setText("")
        self.valueCoef.move(60, 200)

        self.labelIntercept = QtWidgets.QLabel(self)
        self.labelIntercept.setText("Intercept = ")
        self.labelIntercept.move(20, 220)

        self.valueIntercept = QtWidgets.QLabel(self)
        self.valueIntercept.setText("")
        self.valueIntercept.move(80, 220)

        self.labelFormula = QtWidgets.QLabel(self)
        self.labelFormula.setText("Formula = ")
        self.labelFormula.move(120, 250)

        self.valueFormula = QtWidgets.QLabel(self)
        self.valueFormula.setText("")
        self.valueFormula.move(180, 250)

        self.TrainCSVButton = QtWidgets.QPushButton(self)
        self.TrainCSVButton.setText("Open Train Target")
        self.TrainCSVButton.move(0, 0)
        self.TrainCSVButton.clicked.connect(self.TrainCSVButtonClicked)
        self.TrainCSVButton.resize(200, 50)

        self.TrainButton = QtWidgets.QPushButton(self)
        self.TrainButton.setText("TRAIN")
        self.TrainButton.move(0, 110)
        self.TrainButton.clicked.connect(self.TrainButtonClicked)
        self.TrainButton.resize(200, 50)
        self.TrainButton.setEnabled(False)

        self.TestButton = QtWidgets.QPushButton(self)
        self.TestButton.setText("TEST")
        self.TestButton.move(200, 110)
        self.TestButton.clicked.connect(self.TestButtonClicked)
        self.TestButton.resize(200, 50)
        self.TestButton.setEnabled(False)

        self.TestCSVButton = QtWidgets.QPushButton(self)
        self.TestCSVButton.setText("Open Test Target")
        self.TestCSVButton.move(200, 0)
        self.TestCSVButton.resize(200, 50)
        self.TestCSVButton.clicked.connect(self.TestCSVButtonClicked)
        self.TestCSVButton.setEnabled(False)

        self.labelX = QtWidgets.QLabel(self)
        self.labelX.setText("X TARGET")
        self.labelX.move(30, 50)

        self.labelY = QtWidgets.QLabel(self)
        self.labelY.setText("Y TARGET")
        self.labelY.move(125, 50)        
        
        self.labelX = QtWidgets.QLabel(self)
        self.labelX.setText("X TARGET")
        self.labelX.move(230, 50)

        self.labelY = QtWidgets.QLabel(self)
        self.labelY.setText("Y TARGET")
        self.labelY.move(325, 50)

        self.train_X_Target = QComboBox(self)
        self.train_X_Target.move(0, 75)
        self.train_X_Target.setEnabled(False)

        self.train_Y_Target = QComboBox(self)
        self.train_Y_Target.move(100, 75)
        self.train_Y_Target.setEnabled(False)

        self.test_X_Target = QComboBox(self)
        self.test_X_Target.move(200, 75)
        self.test_X_Target.setEnabled(False)

        self.test_Y_Target = QComboBox(self)
        self.test_Y_Target.move(300, 75)
        self.test_Y_Target.setEnabled(False)

    def TrainButtonClicked(self):
        pro.trainModel(str(self.train_X_Target.currentText()), str(self.train_Y_Target.currentText()))
        self.valueIntercept.setText(str(pro.intercept))
        self.valueCoef.setText(str(pro.coef))
        self.valueFormula.setText(str(pro.formula))
        self.TestCSVButton.setEnabled(True),

    def TestButtonClicked(self):
        pro.testModel(str(self.test_X_Target.currentText()), str(self.test_Y_Target.currentText()))
        self.valueMSE.setText(str(pro.MSE))
        self.valueRMSE.setText(str(pro.RMSE))

    def TrainCSVButtonClicked(self):
        self.trainCSVName = QFileDialog.getOpenFileName()[0]
        pro.dataSet = pd.read_csv(self.trainCSVName)
        counter = 0
        self.train_X_Target.clear()
        self.train_Y_Target.clear()
        for i in pro.dataSet:
            if counter != 0:
                self.train_X_Target.addItem(str(i))
                self.train_Y_Target.addItem(str(i))
            else:
                counter += 1
        self.TrainButton.setEnabled(True)
        self.train_X_Target.setEnabled(True)
        self.train_Y_Target.setEnabled(True)
    
    def TestCSVButtonClicked(self):
        self.testCSVName = QFileDialog.getOpenFileName()[0]
        pro.test = pd.read_csv(self.testCSVName)
        counter = 0
        self.test_X_Target.clear()
        self.test_Y_Target.clear()
        for i in pro.dataSet:
            if counter != 0:
                self.test_X_Target.addItem(str(i))
                self.test_Y_Target.addItem(str(i))
            else:
                counter += 1
        self.TestButton.setEnabled(True)
        self.test_X_Target.setEnabled(True)
        self.test_Y_Target.setEnabled(True)
        self.test_X_Target.setCurrentText(self.train_X_Target.currentText())
        self.test_Y_Target.setCurrentText(self.train_Y_Target.currentText())

def window():
    app = QApplication(sys.argv)
    win = GUI(300, 300, 400, 300, "SVR")
    win.show()
    sys.exit(app.exec_())

pro = Product()
window()