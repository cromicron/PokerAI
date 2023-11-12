# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'setupDialoguenyBuUA.ui'
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
        Dialog.resize(441, 264)
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.labelNumPlayers = QLabel(Dialog)
        self.labelNumPlayers.setObjectName(u"labelNumPlayers")
        font = QFont()
        font.setPointSize(10)
        self.labelNumPlayers.setFont(font)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.labelNumPlayers)

        self.spinBoxNumPlayers = QSpinBox(Dialog)
        self.spinBoxNumPlayers.setObjectName(u"spinBoxNumPlayers")
        self.spinBoxNumPlayers.setFont(font)
        self.spinBoxNumPlayers.setMinimum(2)
        self.spinBoxNumPlayers.setMaximum(9)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.spinBoxNumPlayers)

        self.labelStartingStack = QLabel(Dialog)
        self.labelStartingStack.setObjectName(u"labelStartingStack")
        self.labelStartingStack.setFont(font)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.labelStartingStack)

        self.spinBoxStartingStack = QSpinBox(Dialog)
        self.spinBoxStartingStack.setObjectName(u"spinBoxStartingStack")
        self.spinBoxStartingStack.setFont(font)
        self.spinBoxStartingStack.setMinimum(5)
        self.spinBoxStartingStack.setMaximum(100)
        self.spinBoxStartingStack.setValue(100)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.spinBoxStartingStack)

        self.labelUsername = QLabel(Dialog)
        self.labelUsername.setObjectName(u"labelUsername")
        self.labelUsername.setFont(font)

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.labelUsername)

        self.lineEditUsername = QLineEdit(Dialog)
        self.lineEditUsername.setObjectName(u"lineEditUsername")
        self.lineEditUsername.setFont(font)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.lineEditUsername)


        self.verticalLayout.addLayout(self.formLayout)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.okButton = QPushButton(Dialog)
        self.okButton.setObjectName(u"okButton")

        self.horizontalLayout.addWidget(self.okButton)

        self.cancelButton = QPushButton(Dialog)
        self.cancelButton.setObjectName(u"cancelButton")

        self.horizontalLayout.addWidget(self.cancelButton)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.labelNumPlayers.setText(QCoreApplication.translate("Dialog", u"Number of Players", None))
        self.labelStartingStack.setText(QCoreApplication.translate("Dialog", u"Starting Stack", None))
        self.labelUsername.setText(QCoreApplication.translate("Dialog", u"Username", None))
        self.lineEditUsername.setText(QCoreApplication.translate("Dialog", u"cromicron", None))
        self.okButton.setText(QCoreApplication.translate("Dialog", u"Ok", None))
        self.cancelButton.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

