from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QTextEdit, QVBoxLayout, QWidget


class LogInfoTextEdit(QWidget):
    def __init__(self):
        super().__init__()
        self.logTextEdit = QTextEdit()
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.logTextEdit)
        self.vbox = QVBoxLayout()
        self.setStyleSheet('QScrollArea{'
                           'border: None;'
                           '}')
        self.logTextEdit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.vbox.addWidget(self.scrollArea)
        self.setLayout(self.vbox)
        self.initUI()

    def initUI(self):
        self.logTextEdit.resize(self.scrollArea.width() - 18, self.logTextEdit.height())
        self.scrollArea.setWidget(self.logTextEdit)

    def resizeEvent(self, event):
        self.initUI()