# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(895, 219)
        self.SelTrajectorypushButton = QPushButton(Dialog)
        self.SelTrajectorypushButton.setObjectName(u"SelTrajectorypushButton")
        self.SelTrajectorypushButton.setGeometry(QRect(120, 8, 135, 32))
        self.SelTrajectorypushButton.setAutoDefault(False)
        self.TrajectorylineEdit = QLineEdit(Dialog)
        self.TrajectorylineEdit.setObjectName(u"TrajectorylineEdit")
        self.TrajectorylineEdit.setGeometry(QRect(268, 13, 607, 21))
        self.SelSimbNIBSpushButton = QPushButton(Dialog)
        self.SelSimbNIBSpushButton.setObjectName(u"SelSimbNIBSpushButton")
        self.SelSimbNIBSpushButton.setGeometry(QRect(120, 34, 135, 32))
        self.SelSimbNIBSpushButton.setAutoDefault(False)
        self.SimbNIBSlineEdit = QLineEdit(Dialog)
        self.SimbNIBSlineEdit.setObjectName(u"SimbNIBSlineEdit")
        self.SimbNIBSlineEdit.setGeometry(QRect(268, 39, 607, 21))
        self.T1WlineEdit = QLineEdit(Dialog)
        self.T1WlineEdit.setObjectName(u"T1WlineEdit")
        self.T1WlineEdit.setGeometry(QRect(269, 66, 607, 21))
        self.SelT1WpushButton = QPushButton(Dialog)
        self.SelT1WpushButton.setObjectName(u"SelT1WpushButton")
        self.SelT1WpushButton.setGeometry(QRect(121, 61, 135, 32))
        self.SelT1WpushButton.setAutoDefault(False)
        self.SelT1WpushButton.setFlat(False)
        self.ContinuepushButton = QPushButton(Dialog)
        self.ContinuepushButton.setObjectName(u"ContinuepushButton")
        self.ContinuepushButton.setGeometry(QRect(448, 180, 138, 32))
        self.SelTProfilepushButton = QPushButton(Dialog)
        self.SelTProfilepushButton.setObjectName(u"SelTProfilepushButton")
        self.SelTProfilepushButton.setGeometry(QRect(112, 130, 131, 56))
        font = QFont()
        font.setBold(False)
        self.SelTProfilepushButton.setFont(font)
        self.SelTProfilepushButton.setAutoDefault(False)
        self.SelTProfilepushButton.setFlat(False)
        self.ThermalProfilelineEdit = QLineEdit(Dialog)
        self.ThermalProfilelineEdit.setObjectName(u"ThermalProfilelineEdit")
        self.ThermalProfilelineEdit.setGeometry(QRect(270, 150, 607, 21))
        self.CTlineEdit = QLineEdit(Dialog)
        self.CTlineEdit.setObjectName(u"CTlineEdit")
        self.CTlineEdit.setEnabled(False)
        self.CTlineEdit.setGeometry(QRect(268, 100, 607, 21))
        self.CTlineEdit.setCursorPosition(3)
        self.CTlineEdit.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.SelCTpushButton = QPushButton(Dialog)
        self.SelCTpushButton.setObjectName(u"SelCTpushButton")
        self.SelCTpushButton.setEnabled(False)
        self.SelCTpushButton.setGeometry(QRect(170, 95, 84, 30))
        self.SelCTpushButton.setAutoDefault(False)
        self.SelCTpushButton.setFlat(False)
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(23, 101, 58, 16))
        self.CTTypecomboBox = QComboBox(Dialog)
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.setObjectName(u"CTTypecomboBox")
        self.CTTypecomboBox.setGeometry(QRect(80, 96, 82, 30))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.SelTrajectorypushButton.setText(QCoreApplication.translate("Dialog", u"Select Trajectory ...", None))
        self.TrajectorylineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.SelSimbNIBSpushButton.setText(QCoreApplication.translate("Dialog", u"Select SimbNIBS ...", None))
        self.SimbNIBSlineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.T1WlineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.SelT1WpushButton.setText(QCoreApplication.translate("Dialog", u"Select T1W ...", None))
        self.ContinuepushButton.setText(QCoreApplication.translate("Dialog", u"CONTINUE", None))
        self.SelTProfilepushButton.setText(QCoreApplication.translate("Dialog", u"Select Thermal\n"
"profile ...", None))
        self.ThermalProfilelineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.CTlineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.SelCTpushButton.setText(QCoreApplication.translate("Dialog", u"Select", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"Use CT?", None))
        self.CTTypecomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"NO", None))
        self.CTTypecomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"real CT", None))
        self.CTTypecomboBox.setItemText(2, QCoreApplication.translate("Dialog", u"ZTE", None))

    # retranslateUi

