import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QLineEdit, QMainWindow
import sys

class LSVR:
    def __init__(self, csvName):
        self.csvName = csvName
        self.csv = pd.read_csv(self.csvName)
        self.dataSet = self.csv.copy()

    def infoDataset(self):
        print(self.dataSet.head())
        print(self.dataSet.info())
        print(self.dataSet.describe().T)
        print(self.dataSet.corr())

    def selectModel(self, TVCheck, radioCheck, newspaperCheck):
        sales.y = self.dataSet["sales"]
        if TVCheck:
            lm = smf.ols("sales ~ TV", self.dataSet)
            pro.model = lm.fit()
            pro.MSE = mean_squared_error(sales.y, pro.model.fittedvalues)
            pro.RMSE = np.sqrt(mean_squared_error(sales.y, pro.model.fittedvalues))
            pro.x = self.dataSet[["TV"]]
            reg = LinearRegression()
            modelReg = reg.fit(pro.x, sales.y)
            pro.intercept = modelReg.intercept_
            pro.coef = modelReg.coef_
            pro.formula = str(round(pro.model.params[0], 2)) + " + TV" + " * " + str(round(pro.model.params[1], 2))
            g = sns.regplot(pro.x, sales.y, ci = None, label = "TV", scatter_kws = {'color' : 'r', 's' : 9})
            g.set_xlabel("TV Harcamalar")
        if radioCheck:
            lm = smf.ols("sales ~ radio", self.dataSet)
            pro.model = lm.fit()
            pro.MSE = mean_squared_error(sales.y, pro.model.fittedvalues)
            pro.RMSE = np.sqrt(mean_squared_error(sales.y, pro.model.fittedvalues))
            pro.x = self.dataSet[["radio"]]
            reg = LinearRegression()
            modelReg = reg.fit(pro.x, sales.y)
            pro.intercept = modelReg.intercept_
            pro.coef = modelReg.coef_
            pro.formula = str(round(pro.model.params[0], 2)) + " + R" + " * " + str(round(pro.model.params[1], 2))
            g = sns.regplot(pro.x, sales.y, ci = None, label = "Radio", scatter_kws = {'color' : 'b', 's' : 9})
            g.set_xlabel("Radio Harcamalar")
        if newspaperCheck:
            lm = smf.ols("sales ~ newspaper", self.dataSet)
            pro.model = lm.fit()
            pro.MSE = mean_squared_error(sales.y, pro.model.fittedvalues)
            pro.RMSE = np.sqrt(mean_squared_error(sales.y, pro.model.fittedvalues))
            pro.x = self.dataSet[["newspaper"]]
            reg = LinearRegression()
            modelReg = reg.fit(pro.x, sales.y)
            pro.intercept = modelReg.intercept_
            pro.coef = modelReg.coef_
            pro.formula = str(round(pro.model.params[0], 2)) + " + N" + " * " + str(round(pro.model.params[1], 2))
            g = sns.regplot(pro.x, sales.y, ci = None, label = "Newsapaper", scatter_kws = {'color' : 'g', 's' : 9})
            g.set_xlabel("Newspaper Harcamalar")
        if TVCheck == False:
            if radioCheck == False:
                if newspaperCheck == False:
                    print("There is not graph to view!")
        g.set_ylabel("Satış Sayısı")
        plt.legend()
        plt.show()
        

    def compare(self):
        sns.pairplot(self.dataSet, kind = "reg")
        sns.jointplot(x = "TV", y = "sales", data = self.dataSet, kind = "reg")
        sns.jointplot(x = "radio", y = "sales", data = self.dataSet, kind = "reg")
        sns.jointplot(x = "newspaper", y = "sales", data = self.dataSet, kind = "reg")

class Product:
    def __init__(self):
        self.x = None
        self.model = None
        self.MSE = None
        self.RMSE = None
        self.intercept = None
        self.coef = None
        self.formula = ""
    
    def infoModel(self):
        if self.x is not None:
            print("Model Parameters: ", self.model.params)
            print("Model Summarry", self.model.summary().tables[1])
            print("MSE = ", self.MSE)
            print("RMSE = ", self.RMSE)
            print("Intercept = ", self.intercept)
            print("Coef = ", self.coef)
            print(self.model.fittedvalues[0:5])
        else:
            print("No value!")


    def predict(self, value):#linearRegressionBySklearn
        if self.model is not None:
            reg = LinearRegression()
            model = reg.fit(self.x, sales.y)
            return model.predict([[int(value)]])
        else:
            print("No value to predict!")

class Sales:
    def __init__(self):
        self.y = None

class GUI(QMainWindow):
    def __init__(self, x, y, width, height, title):
        super(GUI, self).__init__()
        self.setGeometry(x, y, width, height)
        self.setWindowTitle(title)
        self.initUI()

    def initUI(self):
        
        self.labelMSE = QtWidgets.QLabel(self)
        self.labelMSE.setText("MSE = ")
        self.labelMSE.move(20, 20)

        self.valueMSE = QtWidgets.QLabel(self)
        self.valueMSE.setText("")
        self.valueMSE.move(60, 20)

        self.labelRMSE = QtWidgets.QLabel(self)
        self.labelRMSE.setText("RMSE = ")
        self.labelRMSE.move(20, 40)

        self.valueRMSE = QtWidgets.QLabel(self)
        self.valueRMSE.setText("")
        self.valueRMSE.move(60, 40)

        self.labelCoef = QtWidgets.QLabel(self)
        self.labelCoef.setText("Coef = ")
        self.labelCoef.move(20, 60)

        self.valueCoef = QtWidgets.QLabel(self)
        self.valueCoef.setText("")
        self.valueCoef.move(60, 60)

        self.labelIntercept = QtWidgets.QLabel(self)
        self.labelIntercept.setText("Intercept = ")
        self.labelIntercept.move(20, 80)

        self.valueIntercept = QtWidgets.QLabel(self)
        self.valueIntercept.setText("")
        self.valueIntercept.move(80, 80)

        self.labelFormula = QtWidgets.QLabel(self)
        self.labelFormula.setText("Formula = ")
        self.labelFormula.move(20, 100)

        self.valueFormula = QtWidgets.QLabel(self)
        self.valueFormula.setText("")
        self.valueFormula.move(80, 100)

        self.TVButton = QtWidgets.QPushButton(self)
        self.TVButton.setText("TV ~ Sales")
        self.TVButton.move(0, 0)
        self.TVButton.clicked.connect(self.TVButtonClicked)

        self.RadioButton = QtWidgets.QPushButton(self)
        self.RadioButton.setText("Radio ~ Sales")
        self.RadioButton.move(100, 0)
        self.RadioButton.clicked.connect(self.RadioButtonClicked)

        self.NewspaperButton = QtWidgets.QPushButton(self)
        self.NewspaperButton.setText("Newspaper ~ Sales")
        self.NewspaperButton.move(200, 0)
        self.NewspaperButton.clicked.connect(self.NewspaperButtonClicked)

        self.GettingCFDButton = QtWidgets.QPushButton(self)
        self.GettingCFDButton.setText("Open CFD File")
        self.GettingCFDButton.move(200, 50)
        self.GettingCFDButton.clicked.connect(self.GettingCFDButtonClicked)

        self.PredictButton = QtWidgets.QPushButton(self)
        self.PredictButton.setText("Predict")
        self.PredictButton.move(200, 130)
        self.PredictButton.clicked.connect(self.PredictButtonClicked)

        self.predictValue = QLineEdit(self)
        self.predictValue.setPlaceholderText("Predict the value!")
        self.predictValue.move(20, 130)

        self.valuePrediction = QtWidgets.QLabel(self)
        self.valuePrediction.setText("")
        self.valuePrediction.move(130, 130)

    def TVButtonClicked(self):
        sample = LSVR(self.CFDName[0])
        sample.selectModel(True, False, False)
        self.valueMSE.setText(str(pro.MSE))
        self.valueRMSE.setText(str(pro.RMSE))
        self.valueIntercept.setText(str(pro.intercept))
        self.valueCoef.setText(str(pro.coef))
        self.valueFormula.setText(str(pro.formula))

    def RadioButtonClicked(self):
        sample = LSVR(self.CFDName[0])
        sample.selectModel(False, True, False)        
        self.valueMSE.setText(str(pro.MSE))
        self.valueRMSE.setText(str(pro.RMSE))
        self.valueIntercept.setText(str(pro.intercept))
        self.valueCoef.setText(str(pro.coef))
        self.valueFormula.setText(str(pro.formula))


    def NewspaperButtonClicked(self):
        sample = LSVR(self.CFDName[0])
        sample.selectModel(False, False, True)        
        self.valueMSE.setText(str(pro.MSE))
        self.valueRMSE.setText(str(pro.RMSE))
        self.valueIntercept.setText(str(pro.intercept))
        self.valueCoef.setText(str(pro.coef))
        self.valueFormula.setText(str(pro.formula))


    def PredictButtonClicked(self):
        self.valuePrediction.setText(str(pro.predict(self.predictValue.text()[0])))

    def GettingCFDButtonClicked(self):
        self.CFDName = QFileDialog.getOpenFileName()
        print(self.CFDName[0])

def window():
    app = QApplication(sys.argv)
    win = GUI(300, 300, 500, 200, "Sale Predictor")
    win.show()
    sys.exit(app.exec_())

pro = Product()
sales = Sales()
window()
