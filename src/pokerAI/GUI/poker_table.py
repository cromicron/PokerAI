# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pokerMainWindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1040, 682)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.pokerTable = QGraphicsView(self.centralwidget)
        self.pokerTable.setObjectName(u"pokerTable")
        self.pokerTable.setGeometry(QRect(160, 170, 711, 341))
        palette = QPalette()
        brush = QBrush(QColor(0, 128, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush)
        self.pokerTable.setPalette(palette)
        self.pokerTable.setStyleSheet(u"  background-color: green;\n"
"  border: 2px solid black;\n"
"  border-radius: 100px;")
        self.player0 = QWidget(self.centralwidget)
        self.player0.setObjectName(u"player0")
        self.player0.setGeometry(QRect(460, 520, 131, 72))
        palette1 = QPalette()
        brush1 = QBrush(QColor(46, 206, 255, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette1.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player0.setPalette(palette1)
        self.player0.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout = QVBoxLayout(self.player0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.labelPlayerName_0 = QLabel(self.player0)
        self.labelPlayerName_0.setObjectName(u"labelPlayerName_0")
        font = QFont()
        font.setFamily(u"MS Reference Sans Serif")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.labelPlayerName_0.setFont(font)
        self.labelPlayerName_0.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.labelPlayerName_0.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.labelPlayerName_0)

        self.chipcountPlayer_0 = QLabel(self.player0)
        self.chipcountPlayer_0.setObjectName(u"chipcountPlayer_0")
        self.chipcountPlayer_0.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_0.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.chipcountPlayer_0)

        self.player1 = QWidget(self.centralwidget)
        self.player1.setObjectName(u"player1")
        self.player1.setGeometry(QRect(250, 510, 95, 72))
        palette2 = QPalette()
        palette2.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette2.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette2.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette2.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette2.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette2.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette2.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette2.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette2.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player1.setPalette(palette2)
        self.player1.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_2 = QVBoxLayout(self.player1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.labelPlayerName_1 = QLabel(self.player1)
        self.labelPlayerName_1.setObjectName(u"labelPlayerName_1")
        self.labelPlayerName_1.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_2.addWidget(self.labelPlayerName_1)

        self.chipcountPlayer_1 = QLabel(self.player1)
        self.chipcountPlayer_1.setObjectName(u"chipcountPlayer_1")
        self.chipcountPlayer_1.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_1.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.chipcountPlayer_1)

        self.player2 = QWidget(self.centralwidget)
        self.player2.setObjectName(u"player2")
        self.player2.setGeometry(QRect(70, 380, 95, 72))
        palette3 = QPalette()
        palette3.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette3.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette3.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette3.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette3.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette3.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette3.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette3.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette3.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player2.setPalette(palette3)
        self.player2.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_3 = QVBoxLayout(self.player2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.labelPlayerName_2 = QLabel(self.player2)
        self.labelPlayerName_2.setObjectName(u"labelPlayerName_2")
        self.labelPlayerName_2.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_3.addWidget(self.labelPlayerName_2)

        self.chipcountPlayer_2 = QLabel(self.player2)
        self.chipcountPlayer_2.setObjectName(u"chipcountPlayer_2")
        self.chipcountPlayer_2.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.chipcountPlayer_2)

        self.player3 = QWidget(self.centralwidget)
        self.player3.setObjectName(u"player3")
        self.player3.setGeometry(QRect(70, 230, 95, 72))
        palette4 = QPalette()
        palette4.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette4.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette4.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette4.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette4.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette4.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette4.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette4.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette4.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player3.setPalette(palette4)
        self.player3.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_4 = QVBoxLayout(self.player3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.labelPlayerName_3 = QLabel(self.player3)
        self.labelPlayerName_3.setObjectName(u"labelPlayerName_3")
        self.labelPlayerName_3.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_4.addWidget(self.labelPlayerName_3)

        self.chipcountPlayer_3 = QLabel(self.player3)
        self.chipcountPlayer_3.setObjectName(u"chipcountPlayer_3")
        self.chipcountPlayer_3.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.chipcountPlayer_3)

        self.player4 = QWidget(self.centralwidget)
        self.player4.setObjectName(u"player4")
        self.player4.setGeometry(QRect(290, 100, 95, 72))
        palette5 = QPalette()
        palette5.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette5.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette5.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette5.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette5.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette5.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette5.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette5.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette5.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player4.setPalette(palette5)
        self.player4.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_5 = QVBoxLayout(self.player4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.labelPlayerName_4 = QLabel(self.player4)
        self.labelPlayerName_4.setObjectName(u"labelPlayerName_4")
        self.labelPlayerName_4.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_5.addWidget(self.labelPlayerName_4)

        self.chipcountPlayer_4 = QLabel(self.player4)
        self.chipcountPlayer_4.setObjectName(u"chipcountPlayer_4")
        self.chipcountPlayer_4.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.chipcountPlayer_4)

        self.player5 = QWidget(self.centralwidget)
        self.player5.setObjectName(u"player5")
        self.player5.setGeometry(QRect(510, 100, 95, 72))
        palette6 = QPalette()
        palette6.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette6.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette6.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette6.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette6.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette6.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette6.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette6.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette6.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player5.setPalette(palette6)
        self.player5.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_6 = QVBoxLayout(self.player5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.labelPlayerName_5 = QLabel(self.player5)
        self.labelPlayerName_5.setObjectName(u"labelPlayerName_5")
        self.labelPlayerName_5.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_6.addWidget(self.labelPlayerName_5)

        self.chipcountPlayer_5 = QLabel(self.player5)
        self.chipcountPlayer_5.setObjectName(u"chipcountPlayer_5")
        self.chipcountPlayer_5.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.chipcountPlayer_5)

        self.player6 = QWidget(self.centralwidget)
        self.player6.setObjectName(u"player6")
        self.player6.setGeometry(QRect(710, 100, 95, 72))
        palette7 = QPalette()
        palette7.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette7.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette7.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette7.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette7.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette7.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette7.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette7.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette7.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player6.setPalette(palette7)
        self.player6.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_7 = QVBoxLayout(self.player6)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.labelPlayerName_6 = QLabel(self.player6)
        self.labelPlayerName_6.setObjectName(u"labelPlayerName_6")
        self.labelPlayerName_6.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_7.addWidget(self.labelPlayerName_6)

        self.chipcountPlayer_6 = QLabel(self.player6)
        self.chipcountPlayer_6.setObjectName(u"chipcountPlayer_6")
        self.chipcountPlayer_6.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_6.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.chipcountPlayer_6)

        self.player7 = QWidget(self.centralwidget)
        self.player7.setObjectName(u"player7")
        self.player7.setGeometry(QRect(870, 250, 95, 72))
        palette8 = QPalette()
        palette8.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette8.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette8.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette8.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette8.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette8.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette8.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette8.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette8.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player7.setPalette(palette8)
        self.player7.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_8 = QVBoxLayout(self.player7)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.labelPlayerName_7 = QLabel(self.player7)
        self.labelPlayerName_7.setObjectName(u"labelPlayerName_7")
        self.labelPlayerName_7.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_8.addWidget(self.labelPlayerName_7)

        self.chipcountPlayer_7 = QLabel(self.player7)
        self.chipcountPlayer_7.setObjectName(u"chipcountPlayer_7")
        self.chipcountPlayer_7.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_7.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.chipcountPlayer_7)

        self.player8 = QWidget(self.centralwidget)
        self.player8.setObjectName(u"player8")
        self.player8.setGeometry(QRect(870, 400, 95, 72))
        palette9 = QPalette()
        palette9.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette9.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette9.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette9.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette9.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette9.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette9.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette9.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette9.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.player8.setPalette(palette9)
        self.player8.setStyleSheet(u"background-color: rgb(46, 206, 255);\n"
"font: 14pt \"MS Reference Sans Serif\";\n"
"  border: 2px solid black;\n"
"  border-radius: 20px;")
        self.verticalLayout_9 = QVBoxLayout(self.player8)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.labelPlayerName_8 = QLabel(self.player8)
        self.labelPlayerName_8.setObjectName(u"labelPlayerName_8")
        self.labelPlayerName_8.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")

        self.verticalLayout_9.addWidget(self.labelPlayerName_8)

        self.chipcountPlayer_8 = QLabel(self.player8)
        self.chipcountPlayer_8.setObjectName(u"chipcountPlayer_8")
        self.chipcountPlayer_8.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.chipcountPlayer_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.chipcountPlayer_8)

        self.potLabel = QLabel(self.centralwidget)
        self.potLabel.setObjectName(u"potLabel")
        self.potLabel.setGeometry(QRect(420, 370, 121, 31))
        font1 = QFont()
        font1.setPointSize(14)
        self.potLabel.setFont(font1)
        self.potLabel.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.button0 = QLabel(self.centralwidget)
        self.button0.setObjectName(u"button0")
        self.button0.setGeometry(QRect(580, 470, 28, 28))
        font2 = QFont()
        font2.setFamily(u"MS Reference Sans Serif")
        font2.setPointSize(12)
        font2.setBold(True)
        font2.setWeight(75)
        self.button0.setFont(font2)
        self.button0.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button0.setAlignment(Qt.AlignCenter)
        self.foldButton = QPushButton(self.centralwidget)
        self.foldButton.setObjectName(u"foldButton")
        self.foldButton.setGeometry(QRect(610, 520, 51, 41))
        self.checkCallButton = QPushButton(self.centralwidget)
        self.checkCallButton.setObjectName(u"checkCallButton")
        self.checkCallButton.setGeometry(QRect(670, 520, 51, 41))
        self.betSizeSlider = QSlider(self.centralwidget)
        self.betSizeSlider.setObjectName(u"betSizeSlider")
        self.betSizeSlider.setGeometry(QRect(617, 571, 91, 22))
        self.betSizeSlider.setMinimum(1)
        self.betSizeSlider.setMaximum(100)
        self.betSizeSlider.setOrientation(Qt.Horizontal)
        self.raiseAmountLineEdit = QLineEdit(self.centralwidget)
        self.raiseAmountLineEdit.setObjectName(u"raiseAmountLineEdit")
        self.raiseAmountLineEdit.setGeometry(QRect(710, 566, 71, 31))
        self.betRaiseButton = QPushButton(self.centralwidget)
        self.betRaiseButton.setObjectName(u"betRaiseButton")
        self.betRaiseButton.setGeometry(QRect(730, 520, 51, 41))
        self.holecardP00 = QLabel(self.centralwidget)
        self.holecardP00.setObjectName(u"holecardP00")
        self.holecardP00.setGeometry(QRect(470, 450, 61, 71))
        self.holecardP01 = QLabel(self.centralwidget)
        self.holecardP01.setObjectName(u"holecardP01")
        self.holecardP01.setGeometry(QRect(520, 450, 61, 71))
        self.holecardP10 = QLabel(self.centralwidget)
        self.holecardP10.setObjectName(u"holecardP10")
        self.holecardP10.setGeometry(QRect(240, 447, 61, 71))
        self.holecardP11 = QLabel(self.centralwidget)
        self.holecardP11.setObjectName(u"holecardP11")
        self.holecardP11.setGeometry(QRect(290, 447, 61, 71))
        self.holecardP20 = QLabel(self.centralwidget)
        self.holecardP20.setObjectName(u"holecardP20")
        self.holecardP20.setGeometry(QRect(60, 310, 61, 71))
        self.holecardP21 = QLabel(self.centralwidget)
        self.holecardP21.setObjectName(u"holecardP21")
        self.holecardP21.setGeometry(QRect(110, 310, 61, 71))
        self.holecardP31 = QLabel(self.centralwidget)
        self.holecardP31.setObjectName(u"holecardP31")
        self.holecardP31.setGeometry(QRect(110, 160, 61, 71))
        self.holecardP30 = QLabel(self.centralwidget)
        self.holecardP30.setObjectName(u"holecardP30")
        self.holecardP30.setGeometry(QRect(60, 160, 61, 71))
        self.holecardP40 = QLabel(self.centralwidget)
        self.holecardP40.setObjectName(u"holecardP40")
        self.holecardP40.setGeometry(QRect(330, 30, 61, 71))
        self.holecardP41 = QLabel(self.centralwidget)
        self.holecardP41.setObjectName(u"holecardP41")
        self.holecardP41.setGeometry(QRect(280, 30, 61, 71))
        self.holecardP51 = QLabel(self.centralwidget)
        self.holecardP51.setObjectName(u"holecardP51")
        self.holecardP51.setGeometry(QRect(500, 30, 61, 71))
        self.holecardP50 = QLabel(self.centralwidget)
        self.holecardP50.setObjectName(u"holecardP50")
        self.holecardP50.setGeometry(QRect(550, 30, 61, 71))
        self.holecardP60 = QLabel(self.centralwidget)
        self.holecardP60.setObjectName(u"holecardP60")
        self.holecardP60.setGeometry(QRect(750, 30, 61, 71))
        self.holecardP61 = QLabel(self.centralwidget)
        self.holecardP61.setObjectName(u"holecardP61")
        self.holecardP61.setGeometry(QRect(700, 30, 61, 71))
        self.holecardP70 = QLabel(self.centralwidget)
        self.holecardP70.setObjectName(u"holecardP70")
        self.holecardP70.setGeometry(QRect(860, 180, 61, 71))
        self.holecardP71 = QLabel(self.centralwidget)
        self.holecardP71.setObjectName(u"holecardP71")
        self.holecardP71.setGeometry(QRect(910, 180, 61, 71))
        self.holecardP80 = QLabel(self.centralwidget)
        self.holecardP80.setObjectName(u"holecardP80")
        self.holecardP80.setGeometry(QRect(860, 330, 61, 71))
        self.holecardP81 = QLabel(self.centralwidget)
        self.holecardP81.setObjectName(u"holecardP81")
        self.holecardP81.setGeometry(QRect(910, 330, 61, 71))
        self.button1 = QLabel(self.centralwidget)
        self.button1.setObjectName(u"button1")
        self.button1.setGeometry(QRect(340, 470, 28, 28))
        self.button1.setFont(font2)
        self.button1.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button1.setAlignment(Qt.AlignCenter)
        self.button2 = QLabel(self.centralwidget)
        self.button2.setObjectName(u"button2")
        self.button2.setGeometry(QRect(180, 430, 28, 28))
        self.button2.setFont(font2)
        self.button2.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button2.setAlignment(Qt.AlignCenter)
        self.button3 = QLabel(self.centralwidget)
        self.button3.setObjectName(u"button3")
        self.button3.setGeometry(QRect(180, 210, 28, 28))
        self.button3.setFont(font2)
        self.button3.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button3.setAlignment(Qt.AlignCenter)
        self.button4 = QLabel(self.centralwidget)
        self.button4.setObjectName(u"button4")
        self.button4.setGeometry(QRect(380, 180, 28, 28))
        self.button4.setFont(font2)
        self.button4.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button4.setAlignment(Qt.AlignCenter)
        self.button5 = QLabel(self.centralwidget)
        self.button5.setObjectName(u"button5")
        self.button5.setGeometry(QRect(600, 180, 28, 28))
        self.button5.setFont(font2)
        self.button5.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button5.setAlignment(Qt.AlignCenter)
        self.button6 = QLabel(self.centralwidget)
        self.button6.setObjectName(u"button6")
        self.button6.setGeometry(QRect(790, 180, 28, 28))
        self.button6.setFont(font2)
        self.button6.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button6.setAlignment(Qt.AlignCenter)
        self.button7 = QLabel(self.centralwidget)
        self.button7.setObjectName(u"button7")
        self.button7.setGeometry(QRect(840, 310, 28, 28))
        self.button7.setFont(font2)
        self.button7.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button7.setAlignment(Qt.AlignCenter)
        self.button8 = QLabel(self.centralwidget)
        self.button8.setObjectName(u"button8")
        self.button8.setGeometry(QRect(810, 460, 28, 28))
        self.button8.setFont(font2)
        self.button8.setStyleSheet(u"    border: 2px solid black; /* Black border */\n"
"    border-radius: 12px; /* Half of the width and height to make it round */\n"
"    background-color: white; /* White background or any color you want for the button */\n"
"    min-width: 24px; /* Minimum width for the circle */\n"
"    min-height: 24px; /* Minimum height for the circle */\n"
"    max-width: 24px; /* Maximum width to enforce a circle shape */\n"
"    max-height: 24px; /* Maximum height to enforce a circle shape */")
        self.button8.setAlignment(Qt.AlignCenter)
        self.bet_0 = QLabel(self.centralwidget)
        self.bet_0.setObjectName(u"bet_0")
        self.bet_0.setGeometry(QRect(480, 410, 81, 31))
        self.bet_0.setFont(font1)
        self.bet_0.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_1 = QLabel(self.centralwidget)
        self.bet_1.setObjectName(u"bet_1")
        self.bet_1.setGeometry(QRect(320, 410, 81, 31))
        self.bet_1.setFont(font1)
        self.bet_1.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_2 = QLabel(self.centralwidget)
        self.bet_2.setObjectName(u"bet_2")
        self.bet_2.setGeometry(QRect(220, 380, 81, 31))
        self.bet_2.setFont(font1)
        self.bet_2.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_3 = QLabel(self.centralwidget)
        self.bet_3.setObjectName(u"bet_3")
        self.bet_3.setGeometry(QRect(190, 280, 81, 31))
        self.bet_3.setFont(font1)
        self.bet_3.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_4 = QLabel(self.centralwidget)
        self.bet_4.setObjectName(u"bet_4")
        self.bet_4.setGeometry(QRect(280, 190, 81, 31))
        self.bet_4.setFont(font1)
        self.bet_4.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_5 = QLabel(self.centralwidget)
        self.bet_5.setObjectName(u"bet_5")
        self.bet_5.setGeometry(QRect(510, 190, 81, 31))
        self.bet_5.setFont(font1)
        self.bet_5.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_6 = QLabel(self.centralwidget)
        self.bet_6.setObjectName(u"bet_6")
        self.bet_6.setGeometry(QRect(700, 190, 81, 31))
        self.bet_6.setFont(font1)
        self.bet_6.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_7 = QLabel(self.centralwidget)
        self.bet_7.setObjectName(u"bet_7")
        self.bet_7.setGeometry(QRect(770, 270, 81, 31))
        self.bet_7.setFont(font1)
        self.bet_7.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bet_8 = QLabel(self.centralwidget)
        self.bet_8.setObjectName(u"bet_8")
        self.bet_8.setGeometry(QRect(770, 410, 81, 31))
        self.bet_8.setFont(font1)
        self.bet_8.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(370, 270, 321, 91))
        self.horizontalLayout = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.flop_0 = QLabel(self.layoutWidget)
        self.flop_0.setObjectName(u"flop_0")

        self.horizontalLayout.addWidget(self.flop_0)

        self.flop_1 = QLabel(self.layoutWidget)
        self.flop_1.setObjectName(u"flop_1")

        self.horizontalLayout.addWidget(self.flop_1)

        self.flop_2 = QLabel(self.layoutWidget)
        self.flop_2.setObjectName(u"flop_2")

        self.horizontalLayout.addWidget(self.flop_2)

        self.turn = QLabel(self.layoutWidget)
        self.turn.setObjectName(u"turn")

        self.horizontalLayout.addWidget(self.turn)

        self.river = QLabel(self.layoutWidget)
        self.river.setObjectName(u"river")

        self.horizontalLayout.addWidget(self.river)

        self.buttonNextHand = QPushButton(self.centralwidget)
        self.buttonNextHand.setObjectName(u"buttonNextHand")
        self.buttonNextHand.setGeometry(QRect(70, 560, 101, 41))
        self.check_0 = QLabel(self.centralwidget)
        self.check_0.setObjectName(u"check_0")
        self.check_0.setGeometry(QRect(470, 590, 113, 24))
        self.check_0.setFont(font)
        self.check_0.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_0.setAlignment(Qt.AlignCenter)
        self.check_1 = QLabel(self.centralwidget)
        self.check_1.setObjectName(u"check_1")
        self.check_1.setGeometry(QRect(240, 580, 113, 24))
        self.check_1.setFont(font)
        self.check_1.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_1.setAlignment(Qt.AlignCenter)
        self.check_2 = QLabel(self.centralwidget)
        self.check_2.setObjectName(u"check_2")
        self.check_2.setGeometry(QRect(-20, 400, 113, 24))
        self.check_2.setFont(font)
        self.check_2.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_2.setAlignment(Qt.AlignCenter)
        self.check_3 = QLabel(self.centralwidget)
        self.check_3.setObjectName(u"check_3")
        self.check_3.setGeometry(QRect(-20, 250, 113, 24))
        self.check_3.setFont(font)
        self.check_3.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_3.setAlignment(Qt.AlignCenter)
        self.check_4 = QLabel(self.centralwidget)
        self.check_4.setObjectName(u"check_4")
        self.check_4.setGeometry(QRect(200, 120, 113, 24))
        self.check_4.setFont(font)
        self.check_4.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_4.setAlignment(Qt.AlignCenter)
        self.check_5 = QLabel(self.centralwidget)
        self.check_5.setObjectName(u"check_5")
        self.check_5.setGeometry(QRect(420, 120, 113, 24))
        self.check_5.setFont(font)
        self.check_5.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_5.setAlignment(Qt.AlignCenter)
        self.check_6 = QLabel(self.centralwidget)
        self.check_6.setObjectName(u"check_6")
        self.check_6.setGeometry(QRect(620, 120, 113, 24))
        self.check_6.setFont(font)
        self.check_6.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_6.setAlignment(Qt.AlignCenter)
        self.check_7 = QLabel(self.centralwidget)
        self.check_7.setObjectName(u"check_7")
        self.check_7.setGeometry(QRect(942, 270, 113, 24))
        self.check_7.setFont(font)
        self.check_7.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_7.setAlignment(Qt.AlignCenter)
        self.check_8 = QLabel(self.centralwidget)
        self.check_8.setObjectName(u"check_8")
        self.check_8.setGeometry(QRect(941, 420, 113, 24))
        self.check_8.setFont(font)
        self.check_8.setStyleSheet(u"    background-color: transparent; /* Makes the background of labels transparent */\n"
"    border: none; /* Removes borders from labels if any */")
        self.check_8.setAlignment(Qt.AlignCenter)
        self.scores = QListWidget(self.centralwidget)
        self.scores.setObjectName(u"scores")
        self.scores.setGeometry(QRect(220, 250, 121, 151))
        palette10 = QPalette()
        brush2 = QBrush(QColor(255, 255, 255, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.WindowText, brush2)
        brush3 = QBrush(QColor(0, 85, 0, 255))
        brush3.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.Button, brush3)
        brush4 = QBrush(QColor(0, 127, 0, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.Light, brush4)
        brush5 = QBrush(QColor(0, 106, 0, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.Midlight, brush5)
        brush6 = QBrush(QColor(0, 42, 0, 255))
        brush6.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.Dark, brush6)
        brush7 = QBrush(QColor(0, 56, 0, 255))
        brush7.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.Mid, brush7)
        palette10.setBrush(QPalette.Active, QPalette.Text, brush2)
        palette10.setBrush(QPalette.Active, QPalette.BrightText, brush2)
        palette10.setBrush(QPalette.Active, QPalette.ButtonText, brush2)
        palette10.setBrush(QPalette.Active, QPalette.Base, brush)
        palette10.setBrush(QPalette.Active, QPalette.Window, brush3)
        brush8 = QBrush(QColor(0, 0, 0, 255))
        brush8.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.Shadow, brush8)
        palette10.setBrush(QPalette.Active, QPalette.AlternateBase, brush6)
        brush9 = QBrush(QColor(255, 255, 220, 255))
        brush9.setStyle(Qt.SolidPattern)
        palette10.setBrush(QPalette.Active, QPalette.ToolTipBase, brush9)
        palette10.setBrush(QPalette.Active, QPalette.ToolTipText, brush8)
        brush10 = QBrush(QColor(255, 255, 255, 128))
        brush10.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette10.setBrush(QPalette.Active, QPalette.PlaceholderText, brush10)
#endif
        palette10.setBrush(QPalette.Inactive, QPalette.WindowText, brush2)
        palette10.setBrush(QPalette.Inactive, QPalette.Button, brush3)
        palette10.setBrush(QPalette.Inactive, QPalette.Light, brush4)
        palette10.setBrush(QPalette.Inactive, QPalette.Midlight, brush5)
        palette10.setBrush(QPalette.Inactive, QPalette.Dark, brush6)
        palette10.setBrush(QPalette.Inactive, QPalette.Mid, brush7)
        palette10.setBrush(QPalette.Inactive, QPalette.Text, brush2)
        palette10.setBrush(QPalette.Inactive, QPalette.BrightText, brush2)
        palette10.setBrush(QPalette.Inactive, QPalette.ButtonText, brush2)
        palette10.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette10.setBrush(QPalette.Inactive, QPalette.Window, brush3)
        palette10.setBrush(QPalette.Inactive, QPalette.Shadow, brush8)
        palette10.setBrush(QPalette.Inactive, QPalette.AlternateBase, brush6)
        palette10.setBrush(QPalette.Inactive, QPalette.ToolTipBase, brush9)
        palette10.setBrush(QPalette.Inactive, QPalette.ToolTipText, brush8)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette10.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush10)
#endif
        palette10.setBrush(QPalette.Disabled, QPalette.WindowText, brush6)
        palette10.setBrush(QPalette.Disabled, QPalette.Button, brush3)
        palette10.setBrush(QPalette.Disabled, QPalette.Light, brush4)
        palette10.setBrush(QPalette.Disabled, QPalette.Midlight, brush5)
        palette10.setBrush(QPalette.Disabled, QPalette.Dark, brush6)
        palette10.setBrush(QPalette.Disabled, QPalette.Mid, brush7)
        palette10.setBrush(QPalette.Disabled, QPalette.Text, brush6)
        palette10.setBrush(QPalette.Disabled, QPalette.BrightText, brush2)
        palette10.setBrush(QPalette.Disabled, QPalette.ButtonText, brush6)
        palette10.setBrush(QPalette.Disabled, QPalette.Base, brush3)
        palette10.setBrush(QPalette.Disabled, QPalette.Window, brush3)
        palette10.setBrush(QPalette.Disabled, QPalette.Shadow, brush8)
        palette10.setBrush(QPalette.Disabled, QPalette.AlternateBase, brush3)
        palette10.setBrush(QPalette.Disabled, QPalette.ToolTipBase, brush9)
        palette10.setBrush(QPalette.Disabled, QPalette.ToolTipText, brush8)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette10.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush10)
#endif
        self.scores.setPalette(palette10)
        font3 = QFont()
        font3.setPointSize(10)
        self.scores.setFont(font3)
        self.hud = QWidget(self.centralwidget)
        self.hud.setObjectName(u"hud")
        self.hud.setGeometry(QRect(600, 440, 191, 71))
        self.hud.setFont(font3)
        self.hud.setLayoutDirection(Qt.LeftToRight)
        self.gridLayout = QGridLayout(self.hud)
        self.gridLayout.setObjectName(u"gridLayout")
        self.q_call = QLabel(self.hud)
        self.q_call.setObjectName(u"q_call")
        self.q_call.setLayoutDirection(Qt.LeftToRight)
        self.q_call.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.q_call, 0, 3, 1, 1)

        self.p_fold = QLabel(self.hud)
        self.p_fold.setObjectName(u"p_fold")
        self.p_fold.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.p_fold, 1, 2, 1, 1)

        self.p_bet = QLabel(self.hud)
        self.p_bet.setObjectName(u"p_bet")
        self.p_bet.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.p_bet, 1, 4, 1, 1)

        self.p_call = QLabel(self.hud)
        self.p_call.setObjectName(u"p_call")
        self.p_call.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.p_call, 1, 3, 1, 1)

        self.q_bet = QLabel(self.hud)
        self.q_bet.setObjectName(u"q_bet")
        self.q_bet.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.q_bet, 0, 4, 1, 1)

        self.hideHud = QCheckBox(self.centralwidget)
        self.hideHud.setObjectName(u"hideHud")
        self.hideHud.setGeometry(QRect(630, 420, 51, 21))
        self.hideHud.setChecked(True)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1040, 20))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.labelPlayerName_0.setText(QCoreApplication.translate("MainWindow", u"Player 0", None))
        self.chipcountPlayer_0.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_1.setText(QCoreApplication.translate("MainWindow", u"Player 1", None))
        self.chipcountPlayer_1.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_2.setText(QCoreApplication.translate("MainWindow", u"Player 2", None))
        self.chipcountPlayer_2.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_3.setText(QCoreApplication.translate("MainWindow", u"Player 3", None))
        self.chipcountPlayer_3.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_4.setText(QCoreApplication.translate("MainWindow", u"Player 4", None))
        self.chipcountPlayer_4.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_5.setText(QCoreApplication.translate("MainWindow", u"Player 5", None))
        self.chipcountPlayer_5.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_6.setText(QCoreApplication.translate("MainWindow", u"Player 6", None))
        self.chipcountPlayer_6.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_7.setText(QCoreApplication.translate("MainWindow", u"Player 7", None))
        self.chipcountPlayer_7.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.labelPlayerName_8.setText(QCoreApplication.translate("MainWindow", u"Player 8", None))
        self.chipcountPlayer_8.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.potLabel.setText(QCoreApplication.translate("MainWindow", u"Pot: 0", None))
        self.button0.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.foldButton.setText(QCoreApplication.translate("MainWindow", u"Fold", None))
        self.checkCallButton.setText(QCoreApplication.translate("MainWindow", u"Check", None))
        self.raiseAmountLineEdit.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.betRaiseButton.setText(QCoreApplication.translate("MainWindow", u"Bet", None))
        self.holecardP00.setText("")
        self.holecardP01.setText("")
        self.holecardP10.setText("")
        self.holecardP11.setText("")
        self.holecardP20.setText("")
        self.holecardP21.setText("")
        self.holecardP31.setText("")
        self.holecardP30.setText("")
        self.holecardP40.setText("")
        self.holecardP41.setText("")
        self.holecardP51.setText("")
        self.holecardP50.setText("")
        self.holecardP60.setText("")
        self.holecardP61.setText("")
        self.holecardP70.setText("")
        self.holecardP71.setText("")
        self.holecardP80.setText("")
        self.holecardP81.setText("")
        self.button1.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.button2.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.button3.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.button4.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.button5.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.button6.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.button7.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.button8.setText(QCoreApplication.translate("MainWindow", u"D", None))
        self.bet_0.setText("")
        self.bet_1.setText("")
        self.bet_2.setText("")
        self.bet_3.setText("")
        self.bet_4.setText("")
        self.bet_5.setText("")
        self.bet_6.setText("")
        self.bet_7.setText("")
        self.bet_8.setText("")
        self.flop_0.setText("")
        self.flop_1.setText("")
        self.flop_2.setText("")
        self.turn.setText("")
        self.river.setText("")
        self.buttonNextHand.setText(QCoreApplication.translate("MainWindow", u"Next Hand", None))
        self.check_0.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_1.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_2.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_3.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_4.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_5.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_6.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_7.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.check_8.setText(QCoreApplication.translate("MainWindow", u"Check!", None))
        self.q_call.setText(QCoreApplication.translate("MainWindow", u"7", None))
        self.p_fold.setText(QCoreApplication.translate("MainWindow", u"0.9", None))
        self.p_bet.setText(QCoreApplication.translate("MainWindow", u"0.05", None))
        self.p_call.setText(QCoreApplication.translate("MainWindow", u"0.05", None))
        self.q_bet.setText(QCoreApplication.translate("MainWindow", u"6", None))
        self.hideHud.setText(QCoreApplication.translate("MainWindow", u"show hud", None))
    # retranslateUi

