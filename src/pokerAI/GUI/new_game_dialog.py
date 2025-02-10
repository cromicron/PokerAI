# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'setupDialogueOKrFfK.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(441, 371)
        self.gridLayout = QGridLayout(Dialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.labelUsername = QLabel(Dialog)
        self.labelUsername.setObjectName(u"labelUsername")
        font = QFont()
        font.setPointSize(10)
        self.labelUsername.setFont(font)

        self.gridLayout.addWidget(self.labelUsername, 0, 0, 1, 1)

        self.lineEditUsername = QLineEdit(Dialog)
        self.lineEditUsername.setObjectName(u"lineEditUsername")
        self.lineEditUsername.setFont(font)

        self.gridLayout.addWidget(self.lineEditUsername, 0, 1, 1, 1)

        self.labelNumPlayers = QLabel(Dialog)
        self.labelNumPlayers.setObjectName(u"labelNumPlayers")
        self.labelNumPlayers.setFont(font)

        self.gridLayout.addWidget(self.labelNumPlayers, 1, 0, 1, 1)

        self.spinBoxNumPlayers = QSpinBox(Dialog)
        self.spinBoxNumPlayers.setObjectName(u"spinBoxNumPlayers")
        self.spinBoxNumPlayers.setFont(font)
        self.spinBoxNumPlayers.setMinimum(2)
        self.spinBoxNumPlayers.setMaximum(9)

        self.gridLayout.addWidget(self.spinBoxNumPlayers, 1, 1, 1, 1)

        self.labelStartingStack = QLabel(Dialog)
        self.labelStartingStack.setObjectName(u"labelStartingStack")
        self.labelStartingStack.setFont(font)

        self.gridLayout.addWidget(self.labelStartingStack, 2, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.okButton = QPushButton(Dialog)
        self.okButton.setObjectName(u"okButton")

        self.horizontalLayout.addWidget(self.okButton)

        self.cancelButton = QPushButton(Dialog)
        self.cancelButton.setObjectName(u"cancelButton")

        self.horizontalLayout.addWidget(self.cancelButton)


        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 2)

        self.chipstacksFrame = QFrame(Dialog)
        self.chipstacksFrame.setObjectName(u"chipstacksFrame")
        self.chipstacksFrame.setFrameShape(QFrame.StyledPanel)
        self.chipstacksFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.chipstacksFrame)
        self.verticalLayout.setObjectName(u"verticalLayout")

        self.gridLayout.addWidget(self.chipstacksFrame, 2, 1, 1, 1)


        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.labelUsername.setText(QCoreApplication.translate("Dialog", u"Username", None))
        self.lineEditUsername.setText(QCoreApplication.translate("Dialog", u"cromicron", None))
        self.labelNumPlayers.setText(QCoreApplication.translate("Dialog", u"Number of Players", None))
        self.labelStartingStack.setText(QCoreApplication.translate("Dialog", u"Starting Stacks", None))
        self.okButton.setText(QCoreApplication.translate("Dialog", u"Ok", None))
        self.cancelButton.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

