from PyQt5.QtGui import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from Demo2 import Ui_Dialog2


class Ui_Dialog1(QDialog):
    signal_2 = Signal(list)

    def __init__(self, *args, **kwargs):
        '''
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.title = QLabel()
        self.title.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.userId = QLabel()
        self.userId.setText('UserId: ')
        self.userId.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelName = QLabel()
        self.labelName.setText('Name: ')
        self.labelName.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelNameText = QLineEdit()
        self.labelAge = QLabel()
        self.labelAge.setText('Age: ')
        self.labelAge.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelAgeText = QLineEdit()
        self.labelGender = QLabel()
        self.labelGender.setText('Gender: ')
        self.labelGender.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelGenderText = QLineEdit()
        self.labelShortnessOfBreath = QLabel()
        self.labelShortnessOfBreath.setText('Shortness of Breath: ')
        self.labelShortnessOfBreath.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelShortnessOfBreathText = QLineEdit()
        self.labelWheezing = QLabel()
        self.labelWheezing.setText('Wheezing: ')
        self.labelWheezing.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelWheezingText = QLineEdit()
        self.labelWeightLoss = QLabel()
        self.labelWeightLoss.setText('Weight Loss: ')
        self.labelWeightLoss.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelWeightLossText = QLineEdit()
        self.labelClubbingOfFingerNails = QLabel()
        self.labelClubbingOfFingerNails.setText('Clubbing of Finger Nails: ')
        self.labelClubbingOfFingerNails.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelClubbingOfFingerNailsText = QLineEdit()
        self.labelSnoring = QLabel()
        self.labelSnoring.setText('Snoring: ')
        self.labelSnoring.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelSnoringText = QLineEdit()
        self.labelFatigue = QLabel()
        self.labelFatigue.setText('Fatigue: ')
        self.labelFatigue.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelFatigueText = QLineEdit()
        self.labelSwalEarlyingDifficulty = QLabel()
        self.labelSwalEarlyingDifficulty.setText('SwalEarlying Difficulty: ')
        self.labelSwalEarlyingDifficulty.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelSwalEarlyingDifficultyText = QLineEdit()
        self.labelFrequentCold = QLabel()
        self.labelFrequentCold.setText('Frequent Cold: ')
        self.labelFrequentCold.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelFrequentColdText = QLineEdit()
        self.labelDryCough = QLabel()
        self.labelDryCough.setText('Dry Cough: ')
        self.labelDryCough.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.labelDryCoughText = QLineEdit()
        self.buttonGet = QPushButton()
        self.buttonGet.setText("get")
        self.buttonGet.clicked.connect(self.jump_to_demo2)
        self.buttonPredict = QPushButton()
        self.buttonPredict.setText("predict")
        self.buttonPredict.clicked.connect(self.jump_to_demo1)
        self.main1 = Ui_Dialog2()
        self.signal_2.connect(self.main1.get_data)
        self.grid = QGridLayout()

        self.grid.addWidget(self.userId, 1, 0)
        self.grid.addWidget(self.title, 1, 1)
        self.grid.addWidget(self.labelName, 2, 0)
        self.grid.addWidget(self.labelNameText, 2, 1)
        self.grid.addWidget(self.labelAge, 3, 0)
        self.grid.addWidget(self.labelAgeText, 3, 1)
        self.grid.addWidget(self.labelGender, 4, 0)
        self.grid.addWidget(self.labelGenderText, 4, 1)
        self.grid.addWidget(self.labelShortnessOfBreath, 5, 0)
        self.grid.addWidget(self.labelShortnessOfBreathText, 5, 1)
        self.grid.addWidget(self.labelWheezing, 6, 0)
        self.grid.addWidget(self.labelWheezingText, 6, 1)
        self.grid.addWidget(self.labelWeightLoss, 7, 0)
        self.grid.addWidget(self.labelWeightLossText, 7, 1)
        self.grid.addWidget(self.labelClubbingOfFingerNails, 8, 0)
        self.grid.addWidget(self.labelClubbingOfFingerNailsText, 8, 1)
        self.grid.addWidget(self.labelSnoring, 9, 0)
        self.grid.addWidget(self.labelSnoringText, 9, 1)
        self.grid.addWidget(self.labelFatigue, 10, 0)
        self.grid.addWidget(self.labelFatigueText, 10, 1)
        self.grid.addWidget(self.labelSwalEarlyingDifficulty, 11, 0)
        self.grid.addWidget(self.labelSwalEarlyingDifficultyText, 11, 1)
        self.grid.addWidget(self.labelFrequentCold, 12, 0)
        self.grid.addWidget(self.labelFrequentColdText, 12, 1)
        self.grid.addWidget(self.labelDryCough, 13, 0)
        self.grid.addWidget(self.labelDryCoughText, 13, 1)
        self.grid.addWidget(self.buttonGet, 14, 0)
        self.grid.addWidget(self.buttonPredict, 14, 1)
        self.setLayout(self.grid)
        self.setGeometry(400, 400, 410, 400)
        self.setWindowTitle('Information')
        # self.show()

    def jump_to_demo1(self):
        self.signal_2.emit(
            [self.labelShortnessOfBreathText.text(), self.labelWheezingText.text(), self.labelWeightLossText.text(),
             self.labelClubbingOfFingerNailsText.text(), self.labelSnoringText.text(), self.labelFatigueText.text(), self.labelSwalEarlyingDifficultyText.text(),self.labelFrequentColdText.text(),self.labelDryCoughText.text()])
        self.main1.show()

    def jump_to_demo2(self):
        self.close()

    def get_data(self, para):
        self.title.setText(para)


if __name__ == '__main__':
    app = QApplication([])
    stats = Ui_Dialog1()
    stats.show()
    app.exec_()
