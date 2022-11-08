# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.3.0
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
        Dialog.resize(998, 219)
        self.SelTrajectorypushButton = QPushButton(Dialog)
        self.SelTrajectorypushButton.setObjectName(u"SelTrajectorypushButton")
        self.SelTrajectorypushButton.setGeometry(QRect(235, 8, 135, 32))
        self.SelTrajectorypushButton.setAutoDefault(False)
        self.TrajectorylineEdit = QLineEdit(Dialog)
        self.TrajectorylineEdit.setObjectName(u"TrajectorylineEdit")
        self.TrajectorylineEdit.setGeometry(QRect(380, 13, 607, 21))
        self.SelSimbNIBSpushButton = QPushButton(Dialog)
        self.SelSimbNIBSpushButton.setObjectName(u"SelSimbNIBSpushButton")
        self.SelSimbNIBSpushButton.setGeometry(QRect(235, 34, 135, 32))
        self.SelSimbNIBSpushButton.setAutoDefault(False)
        self.SimbNIBSlineEdit = QLineEdit(Dialog)
        self.SimbNIBSlineEdit.setObjectName(u"SimbNIBSlineEdit")
        self.SimbNIBSlineEdit.setGeometry(QRect(380, 39, 607, 21))
        self.T1WlineEdit = QLineEdit(Dialog)
        self.T1WlineEdit.setObjectName(u"T1WlineEdit")
        self.T1WlineEdit.setGeometry(QRect(381, 66, 607, 21))
        self.SelT1WpushButton = QPushButton(Dialog)
        self.SelT1WpushButton.setObjectName(u"SelT1WpushButton")
        self.SelT1WpushButton.setGeometry(QRect(236, 61, 135, 32))
        self.SelT1WpushButton.setAutoDefault(False)
        self.SelT1WpushButton.setFlat(False)
        self.ContinuepushButton = QPushButton(Dialog)
        self.ContinuepushButton.setObjectName(u"ContinuepushButton")
        self.ContinuepushButton.setGeometry(QRect(459, 180, 239, 32))
        self.SelTProfilepushButton = QPushButton(Dialog)
        self.SelTProfilepushButton.setObjectName(u"SelTProfilepushButton")
        self.SelTProfilepushButton.setGeometry(QRect(227, 130, 131, 56))
        font = QFont()
        font.setBold(False)
        self.SelTProfilepushButton.setFont(font)
        self.SelTProfilepushButton.setAutoDefault(False)
        self.SelTProfilepushButton.setFlat(False)
        self.ThermalProfilelineEdit = QLineEdit(Dialog)
        self.ThermalProfilelineEdit.setObjectName(u"ThermalProfilelineEdit")
        self.ThermalProfilelineEdit.setGeometry(QRect(382, 150, 607, 21))
        self.CTlineEdit = QLineEdit(Dialog)
        self.CTlineEdit.setObjectName(u"CTlineEdit")
        self.CTlineEdit.setEnabled(False)
        self.CTlineEdit.setGeometry(QRect(380, 100, 607, 21))
        self.CTlineEdit.setCursorPosition(3)
        self.CTlineEdit.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.SelCTpushButton = QPushButton(Dialog)
        self.SelCTpushButton.setObjectName(u"SelCTpushButton")
        self.SelCTpushButton.setEnabled(False)
        self.SelCTpushButton.setGeometry(QRect(285, 95, 84, 30))
        self.SelCTpushButton.setAutoDefault(False)
        self.SelCTpushButton.setFlat(False)
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(13, 103, 58, 16))
        self.CTTypecomboBox = QComboBox(Dialog)
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.addItem("")
        self.CTTypecomboBox.setObjectName(u"CTTypecomboBox")
        self.CTTypecomboBox.setGeometry(QRect(60, 98, 82, 30))
        self.SimbNIBSTypecomboBox = QComboBox(Dialog)
        self.SimbNIBSTypecomboBox.addItem("")
        self.SimbNIBSTypecomboBox.addItem("")
        self.SimbNIBSTypecomboBox.setObjectName(u"SimbNIBSTypecomboBox")
        self.SimbNIBSTypecomboBox.setGeometry(QRect(120, 35, 110, 30))
        self.TrajectoryTypecomboBox = QComboBox(Dialog)
        self.TrajectoryTypecomboBox.addItem("")
        self.TrajectoryTypecomboBox.addItem("")
        self.TrajectoryTypecomboBox.setObjectName(u"TrajectoryTypecomboBox")
        self.TrajectoryTypecomboBox.setGeometry(QRect(120, 10, 110, 30))
        self.CoregCTcomboBox = QComboBox(Dialog)
        self.CoregCTcomboBox.addItem("")
        self.CoregCTcomboBox.addItem("")
        self.CoregCTcomboBox.setObjectName(u"CoregCTcomboBox")
        self.CoregCTcomboBox.setEnabled(False)
        self.CoregCTcomboBox.setGeometry(QRect(198, 97, 82, 30))
        self.CoregCTlabel = QLabel(Dialog)
        self.CoregCTlabel.setObjectName(u"CoregCTlabel")
        self.CoregCTlabel.setEnabled(False)
        self.CoregCTlabel.setGeometry(QRect(142, 102, 58, 16))

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

        self.SimbNIBSTypecomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"charm", None))
        self.SimbNIBSTypecomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"headreco", None))

        self.TrajectoryTypecomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"Brainsight", None))
        self.TrajectoryTypecomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"Slicer", None))

        self.CoregCTcomboBox.setItemText(0, QCoreApplication.translate("Dialog", u"NO", None))
        self.CoregCTcomboBox.setItemText(1, QCoreApplication.translate("Dialog", u"CT to MRI", None))

        self.CoregCTlabel.setText(QCoreApplication.translate("Dialog", u"Correg.?", None))
    # retranslateUi

