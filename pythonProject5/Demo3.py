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


# from login import LoginDialog


class Ui_Dialog3(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.label1 = QLabel()
        self.label1.setText('OK!')
        self.label1.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.layout.addRow(self.label1)
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Confirm')
        # self.show()


if __name__ == '__main__':
    app = QApplication([])
    stats = Ui_Dialog3()
    stats.show()
    app.exec_()