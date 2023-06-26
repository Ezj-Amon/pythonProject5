from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from Demo1 import Ui_Dialog1


class LoginDialog(QDialog):
    signal_1 = Signal(str)

    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.layout = QFormLayout()
        self.layout.setFormAlignment(Qt.AlignCenter)
        self.setLayout(self.layout)
        self.label1 = QLabel()
        self.label1.setText('Lung cancer detection system')
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.userId = QLabel('User ID')
        self.password = QLabel('Password')
        self.idEdit = QLineEdit()
        self.passwordEdit = QLineEdit()
        self.button_enter = QPushButton()
        self.button_enter.setText("Login")
        self.layout.addRow(self.label1)
        self.layout.addRow(self.userId, self.idEdit)
        self.layout.addRow(self.password, self.passwordEdit)
        self.layout.addRow(self.button_enter)
        self.button_enter.clicked.connect(self.jump_to_demo1)
        self.main = Ui_Dialog1()
        self.signal_1.connect(self.main.get_data)
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Login')

    def jump_to_demo1(self):
        self.signal_1.emit(self.idEdit.text())
        self.main.show()


if __name__ == "__main__":
    app = QApplication([])
    stats = LoginDialog()
    stats.show()
    app.exec_()
