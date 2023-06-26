from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtUiTools import *
from PySide2.QtGui import *
import pandas as pd
from CNN import predict
from sklearn.preprocessing import MinMaxScaler
from Demo3 import Ui_Dialog3
import numpy as np


# from login import LoginDialog


class Ui_Dialog2(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.label1 = QLabel()
        self.label1.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.label1.move(35, 40)
        self.button1 = QPushButton()
        self.button1.setText("Return to home page")
        self.button1.move(15, 80)
        self.button1.clicked.connect(self.jump_to_login)
        self.label2 = QLabel()
        self.label2.setText('Diagnostic result: ')
        self.label2.setStyleSheet("color:red")
        self.lineE2 = QTextEdit()
        self.button2 = QPushButton()
        self.button2.setText('Confirm')
        self.button2.clicked.connect(self.open_demo3)
        self.layout.addRow(self.label1)
        self.layout.addRow(self.button1)
        self.layout.addRow(self.label2)
        self.layout.addRow(self.lineE2)
        self.layout.addRow(self.button2)
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Result')
        # self.show()

    def jump_to_login(self):
        self.close()

    def open_demo3(self):
        self.second_ui = Ui_Dialog3()
        self.second_ui.show()

    def get_data(self, para):
        print(para)
        para = np.array(list(map(float, para))).reshape(-1, 1)
            # para = np.array([0.125, 0.714285714, 0, 0.142857143, 0.428571429, 0.125, 0, 0.166666667, 0.333333333]).reshape(-1, 1)
        print(para)
        scaler = MinMaxScaler(feature_range=(0, 1))
        pre_data = scaler.fit_transform(para)
        print(pre_data)
        pre_data = pre_data.reshape(pre_data.shape[1], pre_data.shape[0], 1, 1)
        result = predict(pre_data)
        if result[0] == 1:
            self.label1.setText('Early Stage')
            self.lineE2.setText(
                'The patient is in the early stage of lung cancer, it is recommended to review or early treatment to improve the cure rate. Treatment options include surgical resection of the tumor, radiotherapy and chemotherapy.')
        if result[0] == 2:
            self.label1.setText('Middle Stage')
            self.lineE2.setText(
                'The patient was in the middle stage of lung cancer, which typically involves a variety of methods including radiation, surgery and chemotherapy.')
        if result[0] == 3:
            self.label1.setText('Late Stage')
            self.lineE2.setText(
                'The patient was in the late stage of lung cancer.Take analgesic drugs, radiotherapy, chemotherapy, immunotherapy, etc. Although the treatment is difficult, timely and effective treatment and comprehensive support measures can prolong the survival time of patients.')


if __name__ == '__main__':
    app = QApplication([])
    stats = Ui_Dialog2()
    stats.show()
    app.exec_()
