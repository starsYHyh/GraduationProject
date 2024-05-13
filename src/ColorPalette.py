import sys

import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QScrollArea, QHBoxLayout, QApplication, QPushButton, \
    QGridLayout, QGroupBox


class ColorButton(QPushButton):
    """
    颜色按钮
    """

    def __init__(self, color, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color.split(',')
        self.color = [int(self.color[0]), int(self.color[1]), int(self.color[2])]
        self.setFixedSize(40, 40)
        self.setStyleSheet(f"""
            QPushButton {'{'}
                background-color: rgb({color});
                border-radius: 20px;
            {'}'}
            QPushButton:pressed {'{'}

                border: 2px solid black;
                border-radius: 20px;
            {'}'}
        """)
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        ColorPalette.color = self.color


class ColorLabel(QWidget):
    """
    带按钮的颜色标签
    """

    def __init__(self, color, count):
        super().__init__()
        self.colorBtn = ColorButton(color)
        self.countLabel = QLabel(str(count))

        self.countLabel.setFixedHeight(20)
        self.countLabel.setText(str(count))
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.colorBtn)
        self.hbox.addWidget(self.countLabel)
        self.setLayout(self.hbox)
        self.setFixedSize(120, 60)


class ColorPalette(QWidget):
    """
    色盘
    """
    # 已选中的颜色
    color = [255, 0, 0]

    def __init__(self):
        super().__init__()
        self.colorsLabels = None
        self.colorsDict = dict()
        self.columnMaxCount = None
        self.layout = QHBoxLayout()
        self.scroll = QScrollArea()
        self.widget = QGroupBox()
        self.colorsLayout = QGridLayout()
        self.dstColor = [255, 0, 0]

        self.widget.setLayout(self.colorsLayout)
        self.widget.setStyleSheet("QGroupBox { border: None; "
                                  "border: 1px solid grey; "
                                  "border-radius: 10px; "
                                  "background-color: #fff; } ")
        self.scroll.setWidget(self.widget)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea {  "
                                  "border: None; "
                                  "} ")
        self.scroll.viewport().setStyleSheet("background-color: #f0f0f0;"
                                             "border-radius: 10px; ")
        self.layout.addWidget(self.scroll)
        self.setLayout(self.layout)
        self.initUI()
        # print()

    def initUI(self):
        if len(self.colorsDict) == 0:
            return
        self.widget.resize(self.scroll.width() - 100, self.widget.height())
        self.columnMaxCount = (self.scroll.width() - 100) // 120
        i = 0
        for key, elem in self.colorsDict.items():
            y = i // self.columnMaxCount
            x = i % self.columnMaxCount
            self.colorsLayout.addWidget(elem[1], y, x)
            i += 1

    def setColors(self, colorsSrc):
        # 如果输入的是一个路径，则说明是第一次读取颜色，需要对值都赋值为0
        if type(colorsSrc) == str:
            with open(colorsSrc) as colorFile:
                colorsStr = colorFile.readlines()
            colorsDict = dict()
            colors = np.array([[list(map(int, i.strip().split(','))) for i in colorsStr]], dtype=np.uint8)
            self.colorsDict['0, 0, 0'] = [0, ColorLabel('0, 0, 0', 0)]
            self.colorsDict['127, 127, 127'] = [0, ColorLabel('127, 127, 127', 0)]
            self.colorsDict['255, 255, 255'] = [0, ColorLabel('255, 255, 255', 0)]
            colorsDict['0, 0, 0'] = 0
            colorsDict['127, 127, 127'] = 0
            colorsDict['255, 255, 255'] = 0
            for color in colors[0]:
                key = f'{color[0]}, {color[1]}, {color[2]}'
                self.colorsDict[key] = [0, ColorLabel(key, 0)]
                colorsDict[key] = 0
            self.initUI()
            return colorsDict
        # 如果输入的是一个字典，则说明需要根据现有字典来进行更新
        else:
            if len(self.colorsDict) != 0:
                for _, value in self.colorsDict.items():
                    self.colorsLayout.removeWidget(value[1])
                self.colorsDict = dict()
            for key, value in colorsSrc.items():
                self.colorsDict[key] = [len(value), ColorLabel(key, len(value))]
            self.initUI()
            return colorsSrc

    def resizeEvent(self, event):
        self.initUI()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorPalette()
    window.show()
    sys.exit(app.exec())
