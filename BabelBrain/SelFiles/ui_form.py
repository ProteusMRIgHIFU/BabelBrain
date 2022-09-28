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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QLineEdit,
    QPushButton, QSizePolicy, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(800, 219)
        self.SelTrajectorypushButton = QPushButton(Dialog)
        self.SelTrajectorypushButton.setObjectName(u"SelTrajectorypushButton")
        self.SelTrajectorypushButton.setGeometry(QRect(18, 8, 135, 32))
        self.SelTrajectorypushButton.setAutoDefault(False)
        self.TrajectorylineEdit = QLineEdit(Dialog)
        self.TrajectorylineEdit.setObjectName(u"TrajectorylineEdit")
        self.TrajectorylineEdit.setGeometry(QRect(160, 13, 607, 21))
        self.SelSimbNIBSpushButton = QPushButton(Dialog)
        self.SelSimbNIBSpushButton.setObjectName(u"SelSimbNIBSpushButton")
        self.SelSimbNIBSpushButton.setGeometry(QRect(18, 34, 135, 32))
        self.SelSimbNIBSpushButton.setAutoDefault(False)
        self.SimbNIBSlineEdit = QLineEdit(Dialog)
        self.SimbNIBSlineEdit.setObjectName(u"SimbNIBSlineEdit")
        self.SimbNIBSlineEdit.setGeometry(QRect(160, 39, 607, 21))
        self.T1WlineEdit = QLineEdit(Dialog)
        self.T1WlineEdit.setObjectName(u"T1WlineEdit")
        self.T1WlineEdit.setGeometry(QRect(161, 66, 607, 21))
        self.SelT1WpushButton = QPushButton(Dialog)
        self.SelT1WpushButton.setObjectName(u"SelT1WpushButton")
        self.SelT1WpushButton.setGeometry(QRect(19, 61, 135, 32))
        self.SelT1WpushButton.setAutoDefault(False)
        self.SelT1WpushButton.setFlat(False)
        self.ContinuepushButton = QPushButton(Dialog)
        self.ContinuepushButton.setObjectName(u"ContinuepushButton")
        self.ContinuepushButton.setGeometry(QRect(340, 180, 138, 32))
        self.SelTProfilepushButton = QPushButton(Dialog)
        self.SelTProfilepushButton.setObjectName(u"SelTProfilepushButton")
        self.SelTProfilepushButton.setGeometry(QRect(20, 130, 135, 56))
        font = QFont()
        font.setBold(True)
        self.SelTProfilepushButton.setFont(font)
        self.SelTProfilepushButton.setAutoDefault(False)
        self.SelTProfilepushButton.setFlat(False)
        self.ThermalProfilelineEdit = QLineEdit(Dialog)
        self.ThermalProfilelineEdit.setObjectName(u"ThermalProfilelineEdit")
        self.ThermalProfilelineEdit.setGeometry(QRect(162, 150, 607, 21))
        self.CTcheckBox = QCheckBox(Dialog)
        self.CTcheckBox.setObjectName(u"CTcheckBox")
        self.CTcheckBox.setGeometry(QRect(16, 98, 85, 20))
        self.CTlineEdit = QLineEdit(Dialog)
        self.CTlineEdit.setObjectName(u"CTlineEdit")
        self.CTlineEdit.setEnabled(False)
        self.CTlineEdit.setGeometry(QRect(160, 100, 607, 21))
        self.CTlineEdit.setCursorPosition(3)
        self.CTlineEdit.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.SelCTpushButton = QPushButton(Dialog)
        self.SelCTpushButton.setObjectName(u"SelCTpushButton")
        self.SelCTpushButton.setEnabled(False)
        self.SelCTpushButton.setGeometry(QRect(84, 95, 71, 32))
        self.SelCTpushButton.setAutoDefault(False)
        self.SelCTpushButton.setFlat(False)

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
        self.CTcheckBox.setText(QCoreApplication.translate("Dialog", u"Use CT", None))
        self.CTlineEdit.setText(QCoreApplication.translate("Dialog", u"...", None))
        self.SelCTpushButton.setText(QCoreApplication.translate("Dialog", u"Select", None))
    # retranslateUi

