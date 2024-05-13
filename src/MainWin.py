import cv2
import sys
import os
import time

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout, QFileDialog, QPushButton, QMessageBox, \
    QHBoxLayout, QMainWindow, QTabWidget, QGridLayout, QGroupBox, QComboBox, QSpinBox, QSplitter
from PyQt6.QtGui import QIcon, QPainter, QColor
from functools import partial

from DenoiseAndPixel import DenoiseAndPixel
from ColorIntroduction import ColorIntroduction
from RoundFill import RoundFill
from Helper import BinSearch, exportToExcel, exportToCsv
from MScrollLabel import MScrollLabel
from ColorPalette import ColorPalette
from LogInfoTextEdit import LogInfoTextEdit


class Main(QMainWindow):
    def __init__(self):
        # 标签页、原图像页、处理后的图像页、原图像、处理后的图像
        super().__init__()
        self.dotSize = 10

        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.srcSL = None
        self.dstSL = None
        self.resSL = None
        # bgr
        self.bgColor = (0, 0, 0)
        self.srcImg = None
        self.dstImg = None
        self.resImg = None
        self.colorsInfo = None
        self.colorsDict = None
        self.mouseColor = None
        self.mousePos = [-1, -1]
        self.start = [-1, -1]
        self.end = [-1, -1]
        self.isSelecting = False
        self.dstColor = [255, 0, 0]
        self.colorsPath = ''
        self.denoisePlan = 0
        self.introductionPlan = 0
        self.log = ''

        self.tabWidget = QTabWidget()
        self.selectImgBtn = QPushButton('Choose Image')
        self.handlerImgBtn = QPushButton('Handle Image')
        self.getresImgBtn = QPushButton('Get Result')
        self.setDotSizeSb = QSpinBox()
        self.setDenoisePlanCb = QComboBox()
        self.setInroductionPlanCb = QComboBox()
        self.setBgColorCb = QComboBox()
        self.exportImgBtn = QPushButton('Export image')
        self.importColorsBtn = QPushButton('Generate chromatogram')
        self.getMouseColorBtn = QPushButton('Get mouse info')
        self.sameColorBtn = QPushButton('Same color change')
        self.pointColorBtn = QPushButton('Point color change')
        self.getRegionBtn = QPushButton('Select region')
        self.areaColorBtn = QPushButton('Area color change')
        self.colorPalette = ColorPalette()
        self.exportLogBtn = QPushButton('Export log')
        self.exportColorsBtn = QPushButton('Export colors')

        self.infoTextEdit = LogInfoTextEdit()
        self.initUI()

    def initUI(self):
        """
        初始化UI
        :return:
        """
        # 窗口设置
        self.resize(1200, 600)
        self.setWindowTitle('Diamond Painting Design')
        self.setWindowIcon(QIcon('tuyuan.png'))

        # 标签页属性设置
        self.srcSL = MScrollLabel(h=600, w=800)
        self.dstSL = MScrollLabel(h=600, w=800)
        self.dstSL.label.paintEvent = self.paintEvt
        self.dstSL.label.mousePressEvent = self.mousePressDefault
        self.dstSL.label.mouseReleaseEvent = self.mouseReleaseDefault
        self.resSL = MScrollLabel(h=600, w=800)

        # 用户输入图像、色谱
        self.selectImgBtn.setToolTip('Choose image from local disk')
        self.selectImgBtn.setFixedSize(self.selectImgBtn.sizeHint())
        self.selectImgBtn.clicked.connect(self.chooseImg)
        self.importColorsBtn.setToolTip('Import colors from local disk')
        self.importColorsBtn.setFixedSize(self.importColorsBtn.sizeHint())
        self.importColorsBtn.clicked.connect(partial(self.updateColors, 0))

        # 调整点大小、处理方案、处理图像
        self.setDotSizeSb.setToolTip('Set dot size')
        self.setDotSizeSb.setFixedSize(60, 26)
        self.setDotSizeSb.setValue(self.dotSize)
        self.setDotSizeSb.setMinimum(6)
        self.setDotSizeSb.setMaximum(20)
        self.setDotSizeSb.valueChanged.connect(self.setDotSize)
        self.setDenoisePlanCb.setToolTip('Denoise plan')
        self.setDenoisePlanCb.addItems(['Default', 'Mean', 'Bilateral', 'NLM'])
        self.setDenoisePlanCb.setCurrentIndex(0)
        self.setDenoisePlanCb.setFixedSize(self.setDenoisePlanCb.sizeHint())
        self.setDenoisePlanCb.currentIndexChanged.connect(self.setDenoisePlan)
        self.setInroductionPlanCb.setToolTip('Introduction plan')
        self.setInroductionPlanCb.addItems(['Default', 'Speed', 'Quality'])
        self.setInroductionPlanCb.setCurrentIndex(0)
        self.setInroductionPlanCb.setFixedSize(self.setInroductionPlanCb.sizeHint())
        self.setInroductionPlanCb.currentIndexChanged.connect(self.setIntroductionPlan)
        self.handlerImgBtn.setFixedSize(self.selectImgBtn.sizeHint())
        self.handlerImgBtn.clicked.connect(self.handleImg)

        # 阵列图像、导出图像
        self.setBgColorCb.setToolTip('Set background color')
        self.setBgColorCb.addItems(['Black', 'Gray', 'White'])
        self.setBgColorCb.setCurrentIndex(0)
        self.setBgColorCb.setFixedSize(self.setBgColorCb.sizeHint())
        self.setBgColorCb.currentIndexChanged.connect(self.setBgColor)
        self.getresImgBtn.setToolTip('Get result image')
        self.getresImgBtn.setFixedSize(self.getresImgBtn.sizeHint())
        self.getresImgBtn.clicked.connect(self.setResImg)

        # 日志
        self.infoTextEdit.logTextEdit.setReadOnly(True)
        self.infoTextEdit.logTextEdit.setStyleSheet('background-color: rgb(255, 255, 255);'
                                                    'border: 1px solid grey;'
                                                    'border-radius: 5px;')

        # 导出
        self.exportImgBtn.setToolTip('Export image')
        self.exportImgBtn.setFixedSize(self.exportImgBtn.sizeHint())
        self.exportImgBtn.clicked.connect(self.exportImg)
        self.exportLogBtn.setToolTip('Export log to txt')
        self.exportLogBtn.setFixedSize(self.exportLogBtn.sizeHint())
        self.exportLogBtn.clicked.connect(self.exportLog)
        self.exportColorsBtn.setToolTip('Export colors to excel or csv')
        self.exportColorsBtn.setFixedSize(self.exportColorsBtn.sizeHint())
        self.exportColorsBtn.clicked.connect(self.exportColors)
        # 颜色替换
        self.getMouseColorBtn.setToolTip('Get information of mouse')
        self.getMouseColorBtn.setFixedSize(self.getMouseColorBtn.sizeHint())
        self.getMouseColorBtn.clicked.connect(self.getMouseColor)
        self.sameColorBtn.setToolTip('Same color change, left click to dstColor, right click to bgColor')
        self.sameColorBtn.setFixedSize(self.sameColorBtn.sizeHint())
        self.sameColorBtn.mousePressEvent = self.sameColorChange
        self.pointColorBtn.setToolTip('Point color change, left click to dstColor, right click to bgColor')
        self.pointColorBtn.setFixedSize(self.pointColorBtn.sizeHint())
        self.pointColorBtn.mousePressEvent = self.pointColorChange
        self.getRegionBtn.setToolTip('Select region')
        self.getRegionBtn.setFixedSize(self.getRegionBtn.sizeHint())
        self.getRegionBtn.clicked.connect(self.getRegionColor)
        self.areaColorBtn.setToolTip('Area color change, left click to dstColor, right click to bgColor')
        self.areaColorBtn.setFixedSize(self.areaColorBtn.sizeHint())
        self.areaColorBtn.mousePressEvent = self.areaColorChange

        # 图像展示区域
        # 为了将图像能在标签页中居中显示
        srcSlW = QWidget()
        srcSlHb = QVBoxLayout()
        srcSlHb.addWidget(self.srcSL)
        srcSlW.setLayout(srcSlHb)
        self.tabWidget.addTab(srcSlW, 'srcImage')

        dstSlW = QWidget()
        dstSlHb = QVBoxLayout()
        dstSlHb.addWidget(self.dstSL)
        dstSlW.setLayout(dstSlHb)
        self.tabWidget.addTab(dstSlW, 'dstImage')

        resSlW = QWidget()
        resSlHb = QVBoxLayout()
        resSlHb.addWidget(self.resSL)
        resSlW.setLayout(resSlHb)
        self.tabWidget.addTab(resSlW, 'resImage')

        # 操作及信息区域
        vbox = QVBoxLayout()

        # 用户输入图像和色谱
        h1 = QHBoxLayout()
        h1.addWidget(self.selectImgBtn)
        h1.addWidget(self.importColorsBtn)
        vbox.addLayout(h1)

        # 调整点大小、处理方案
        h2 = QHBoxLayout()
        h2.addWidget(self.handlerImgBtn)
        h2.addWidget(self.setDotSizeSb)
        h2.addWidget(self.setDenoisePlanCb)
        h2.addWidget(self.setInroductionPlanCb)
        vbox.addLayout(h2)

        # 单点换色、同色换色
        gb1 = QGroupBox()
        g = QGridLayout()
        g.addWidget(self.getMouseColorBtn, 0, 0, 2, 1, alignment=Qt.AlignmentFlag.AlignVCenter)
        g.addWidget(self.sameColorBtn, 0, 1)
        g.addWidget(self.pointColorBtn, 1, 1)
        gb1.setLayout(g)
        gb1.setStyleSheet("QGroupBox { border: 1px solid grey; "
                          "border-radius: 5px; "
                          "background-color: #f0f0f0; } "
                          "QGroupBox::title { subcontrol-origin: margin; "
                          "left: 10px; padding: 0 5px 0 5px; } ")
        vbox.addWidget(gb1)

        # 区域换色
        h3 = QHBoxLayout()
        h3.addWidget(self.getRegionBtn)
        h3.addWidget(self.areaColorBtn)
        vbox.addLayout(h3)

        # 生成圆点图像
        h4 = QHBoxLayout()
        h4.addWidget(self.setBgColorCb)
        h4.addWidget(self.getresImgBtn)
        vbox.addLayout(h4)

        # 色谱展示、日志展示
        pltContainer = QWidget()
        v0 = QVBoxLayout()
        spl1 = QSplitter(Qt.Orientation.Vertical)
        spl1.addWidget(self.colorPalette)
        spl1.addWidget(self.infoTextEdit)
        spl1.setSizes([200, 100])
        spl1.setStyleSheet("QSplitter::handle::hover { background-color: darkgray }")
        v0.addWidget(spl1)
        pltContainer.setLayout(v0)
        vbox.addWidget(pltContainer)

        # 导出相关内容
        h5 = QHBoxLayout()
        h5.addWidget(self.exportColorsBtn, 1)
        h5.addWidget(self.exportLogBtn, 1)
        h5.addWidget(self.exportImgBtn, 1)
        vbox.addLayout(h5)

        # 总体的布局
        optContainer = QWidget()
        optContainer.setLayout(vbox)

        spl2 = QSplitter(Qt.Orientation.Horizontal)
        spl2.addWidget(self.tabWidget)
        spl2.setSizes([400, 200])
        spl2.addWidget(optContainer)

        hContainer = QHBoxLayout()
        hContainer.addWidget(spl2)
        self.mainWidget.setLayout(hContainer)

        self.statusBar().showMessage('Ready!')
        self.updateLog('Ready!')

        self.show()

    def getMouseColor(self):
        self.dstSL.label.mousePressEvent = self.mousePressClick
        self.dstSL.label.mouseReleaseEvent = self.mouseReleaseDefault

    def getRegionColor(self):
        self.isSelecting = True
        self.dstSL.label.mousePressEvent = self.mousePressSelect
        self.dstSL.label.mouseReleaseEvent = self.mouseReleaseSelect
        if self.dstSL.label.mouseReleaseEvent == self.mouseReleaseSelect:
            print('ReleaseSelect')

    def mousePressClick(self, event):
        """在点击情况下的Label点击事件"""
        imgHeight, imgWidth, _ = self.dstImg.shape
        x = int(event.pos().x() / self.dotSize) * self.dotSize
        y = int(event.pos().y() / self.dotSize) * self.dotSize
        self.mousePos = [y, x]
        if 0 <= x < imgWidth and 0 <= y < imgHeight:
            self.mouseColor = self.dstImg[y][x]
        self.updateLog(
            f'User select a pixel in the image, and mouse position is {self.mousePos} and mouse color is {self.mouseColor}')

    def mousePressDefault(self, event):
        """在普通情况下的Label点击事件"""
        return

    def mousePressSelect(self, event):
        """选择区域时的Label点击事件"""
        self.start = [event.pos().y(), event.pos().x()]
        print(f'self.start = {self.start}')

    def mouseReleaseSelect(self, event):
        self.end = [event.pos().y(), event.pos().x()]
        print(f'self.end = {self.end}')
        self.updateLog(f'User select a region in the image, and region is {self.start} to {self.end}')
        self.dstSL.label.update()

    def mouseReleaseDefault(self, event):
        return

    def paintEvt(self, event):
        self.dstSL.label.paintEvt(event)
        if self.isSelecting:
            print(self.start)
            print(self.end)
            painter = QPainter(self.dstSL.label)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            startX = min(self.start[1], self.end[1])
            startY = min(self.start[0], self.end[0])
            endX = max(self.start[1], self.end[1])
            endY = max(self.start[0], self.end[0])
            painter.setPen(QColor(0, 0, 255))
            painter.drawRect(startX, startY, endX - startX, endY - startY)
            painter.end()

    def sameColorChange(self, e):
        """
        同色换色
        :return:
        """
        if -1 in self.mousePos:
            self.statusBar().showMessage('Wrong mouse position!')
            self.updateLog('User try to change same color, but mouse position is wrong')
            return
        self.dstSL.label.mousePressEvent = self.mousePressDefault
        self.dstColor = ColorPalette.color
        key = self.getDictKey([self.mousePos[0], self.mousePos[1]])
        # 如果是左键，就换成目标颜色，如果是右键，就换成背景颜色
        if e.buttons() == Qt.MouseButton.LeftButton:
            # 如果目标颜色和背景颜色相同，弹出警告框
            if key == f'{self.bgColor[0]}, {self.bgColor[1]}, {self.bgColor[2]}':
                messageBox = QMessageBox()
                messageBox.setWindowTitle('Warning')
                messageBox.setText('You are trying change background color to dstColor, are you sure?')
                messageBox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                messageBox.setDefaultButton(QMessageBox.StandardButton.No)
                # 弹出对话框并获取用户的选择
                btnClicked = messageBox.exec()

                # 处理用户的选择
                if btnClicked == QMessageBox.StandardButton.No:
                    self.updateLog('User try to change background color to dstColor, but user cancel it')
                    self.mousePos = [-1, -1]
                    return
                elif btnClicked == QMessageBox.StandardButton.Yes:
                    self.updateLog('User try to change background color to dstColor, and user confirm it')
                    pass
            dstColor = self.dstColor
            dstColorStr = f'{self.dstColor[0]}, {self.dstColor[1]}, {self.dstColor[2]}'
        elif e.buttons() == Qt.MouseButton.RightButton:
            dstColor = self.bgColor
            dstColorStr = f'{self.bgColor[0]}, {self.bgColor[1]}, {self.bgColor[2]}'
        else:
            return

        if key == dstColorStr:
            self.mousePos = [-1, -1]
            return

        points = self.colorsDict[key]
        imgHeight, imgWidth, _ = self.dstImg.shape
        y, x = self.mousePos
        key = self.getDictKey([y, x])
        self.colorsDict[dstColorStr].extend(self.colorsDict[key])

        self.colorPalette.colorsDict[key][0] = 0
        self.colorPalette.colorsDict[key][1].countLabel.setText(str(self.colorPalette.colorsDict[key][0]))
        self.colorPalette.colorsDict[dstColorStr][0] += len(points)
        self.colorPalette.colorsDict[dstColorStr][1].countLabel.setText(
            str(self.colorPalette.colorsDict[dstColorStr][0]))

        # 清空旧颜色列表
        self.colorsDict[key] = []

        dotSize = self.dotSize
        self.updateLog(f'User try to change same color, and mouse position is {self.mousePos}'
                       f' and the color is {dstColor}')
        tStart = time.time()

        # 优化，0.044 -> 0.001
        for i, (y, x) in enumerate(points):
            y1, y2 = y, min(y + dotSize, imgHeight)
            x1, x2 = x, min(x + dotSize, imgWidth)
            self.dstImg[y1: y2, x1: x2] = dstColor[::-1]

        tEnd = time.time()
        self.statusBar().showMessage(f'The replacement was successful, a total of {len(points)} grids were replaced, \
                                    time cost is {tEnd - tStart}s.')
        self.dstSL.setImg(self.dstImg)
        self.updateLog(f'The replacement was successful, a total of {len(points)} grids were replaced,'
                       f' time cost is {tEnd - tStart}s.')
        self.mousePos = [-1, -1]

    def areaColorChange(self, e):
        """
        区域换色
        :return:
        """
        if -1 in self.start or -1 in self.end:
            self.statusBar().showMessage('Wrong region position!')
            self.updateLog('User try to change area color, but region position is wrong')
            return
        imHeight, imWidth, _ = self.dstImg.shape
        self.dstSL.label.update()
        self.dstSL.label.mousePressEvent = self.mousePressDefault
        self.dstSL.label.mouseReleaseEvent = self.mouseReleaseDefault

        self.isSelecting = False
        startX = int(max(min(self.start[1], self.end[1]), 0) / self.dotSize) * self.dotSize
        startY = int(max(min(self.start[0], self.end[0]), 0) / self.dotSize) * self.dotSize
        endX = int(min(max(self.start[1], self.end[1]), imWidth - 1) / self.dotSize) * self.dotSize
        endY = int(min(max(self.start[0], self.end[0]), imHeight - 1) / self.dotSize) * self.dotSize
        self.start = [startY, startX]
        self.end = [endY, endX]

        self.dstColor = ColorPalette.color
        self.updateLog(f'User try to change area color, and region is {self.start} to {self.end}'
                       f' and the color is {self.dstColor}')

        if e.buttons() == Qt.MouseButton.LeftButton:
            dstColor = self.dstColor
        elif e.buttons() == Qt.MouseButton.RightButton:
            dstColor = self.bgColor
        else:
            return
        dstColorStr = f'{dstColor[0]}, {dstColor[1]}, {dstColor[2]}'

        count = 0

        # 0.11 -> 0.011
        tStart = time.time()
        for i in range(startY, endY + self.dotSize, self.dotSize):
            for j in range(startX, endX + self.dotSize, self.dotSize):
                # 进行数据的更新
                key = self.getDictKey([i, j])
                if key == dstColorStr:
                    continue
                positions = self.colorsDict[key]
                # 二分查找索引
                index = BinSearch(positions, [i, j], imWidth)
                count += 1
                # 更新图像
                self.dstImg[i: i + self.dotSize, j: j + self.dotSize] = dstColor[::-1]
                # 将该点从旧颜色地点列表中删除
                self.colorsDict[key].pop(index)
                # 将该点添加到新颜色地点列表中
                self.colorsDict[dstColorStr].append([i, j])
                self.colorPalette.colorsDict[key][0] -= 1
                self.colorPalette.colorsDict[key][1].countLabel.setText(str(self.colorPalette.colorsDict[key][0]))
                self.colorPalette.colorsDict[dstColorStr][0] += 1
                self.colorPalette.colorsDict[dstColorStr][1].countLabel.setText(
                    str(self.colorPalette.colorsDict[dstColorStr][0]))
        # 将目标颜色对应的位置进行重新排序
        self.colorsDict[dstColorStr] = sorted(self.colorsDict[dstColorStr], key=lambda x: (x[0], x[1]))
        tEnd = time.time()
        self.statusBar().showMessage(f'The replacement was successful, a total of {count} grids were replaced, cost = '
                                     + f'{tEnd - tStart}s')

        self.dstSL.setImg(self.dstImg)
        self.updateLog(f'The replacement was successful, a total of {count} grids were replaced, cost = '
                       f'{tEnd - tStart}s')
        self.start = [-1, -1]
        self.end = [-1, -1]

    def pointColorChange(self, e):
        """
        单点换色
        :return:
        """
        if -1 in self.mousePos:
            self.statusBar().showMessage('Wrong mouse position!')
            self.updateLog('User try to change point color, but mouse position is wrong')
            return
        self.dstSL.label.mousePressEvent = self.mousePressDefault
        imgHeight, imgWidth, _ = self.dstImg.shape
        x = self.mousePos[1]
        y = self.mousePos[0]
        # 进行数据的更新
        self.dstColor = ColorPalette.color

        if e.buttons() == Qt.MouseButton.LeftButton:
            dstColor = self.dstColor
            self.updateLog(f'User try to change point color, and position is {self.mousePos}'
                           f' and the color is {self.dstColor}')
        elif e.buttons() == Qt.MouseButton.RightButton:
            dstColor = self.bgColor
            self.updateLog(f'User tries to replace the color of the point at position {self.mousePos} '
                           f'with the background color')
        else:
            return

        dstColorStr = f'{dstColor[0]}, {dstColor[1]}, {dstColor[2]}'
        key = self.getDictKey([y, x])
        if key == dstColorStr:
            self.mousePos = [-1, -1]
            return
        index = BinSearch(self.colorsDict[key], position=[y, x], width=imgWidth)
        # 在旧颜色删除位置
        self.colorsDict[key].pop(index)
        self.colorPalette.colorsDict[key][0] -= 1
        self.colorPalette.colorsDict[key][1].countLabel.setText(
            str(self.colorPalette.colorsDict[key][0]))
        # 在新颜色添加位置
        self.colorsDict[dstColorStr].append([y, x])
        self.colorPalette.colorsDict[dstColorStr][0] += 1
        self.colorPalette.colorsDict[dstColorStr][1].countLabel.setText(
            str(self.colorPalette.colorsDict[dstColorStr][0]))
        self.colorsDict[dstColorStr] = sorted(self.colorsDict[dstColorStr], key=lambda x: (x[0], x[1]))
        self.dstImg[y:min(y + self.dotSize, imgHeight), x:min(x + self.dotSize, imgWidth)] = dstColor[::-1]

        self.statusBar().showMessage(f'The replacement was successful, a total of 1 grid were replaced')
        self.dstSL.setImg(self.dstImg)
        self.updateLog(f'The replacement was successful, a total of 1 grid were replaced')
        self.mousePos = [-1, -1]

    def getDictKey(self, pos):
        """
        根据位置获取颜色字典的key
        :param pos:
        :return:
        """
        [b, g, r] = self.dstImg[pos[0], pos[1]]
        return f'{r}, {g}, {b}'

    def chooseImg(self):
        """
        从文件夹选择图片
        :return:
        """
        imgPath, _ = QFileDialog.getOpenFileName(None, 'Choose an image', '', 'Image file(*.jpg *.png *.bmp)')
        if len(imgPath) != 0:
            # 解决读取无法读取中文路径的问题
            self.srcImg = cv2.imdecode(np.fromfile(file=imgPath, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.srcSL.setImg(self.srcImg)
            self.dstImg = np.full(self.srcImg.shape, self.bgColor, dtype=np.uint8)
            self.resImg = np.full(self.srcImg.shape, self.bgColor, dtype=np.uint8)
            self.dstSL.setImg(self.dstImg)
            self.resSL.setImg(self.resImg)
            self.statusBar().showMessage('Image imported successfully!')
            self.updateLog(f'User import an image from path {imgPath}\n' + 'Image imported successfully!')

    def setDotSize(self, value):
        """
        设置点的大小
        :param value: 点的大小
        :return:
        """
        self.dotSize = value
        self.updateLog(f'User set dot size to {self.dotSize}')

    def setDenoisePlan(self, index):
        """
        设置去噪方案
        :param index: 方案索引
        :return:
        """
        self.denoisePlan = index
        self.updateLog(f'User set denoise plan to {self.setDenoisePlanCb.itemText(self.denoisePlan)}')

    def setIntroductionPlan(self, index):
        """
        设置色彩归纳方案
        :param index:
        :return:
        """
        self.introductionPlan = index
        self.updateLog(f'User set introduction plan to {self.setInroductionPlanCb.itemText(self.introductionPlan)}')

    def setBgColor(self, index):
        """
        设置背景颜色
        :param index:
        :return:
        """
        if index == 0:
            self.bgColor = (0, 0, 0)
        elif index == 1:
            self.bgColor = (127, 127, 127)
        else:
            self.bgColor = (255, 255, 255)
        self.updateLog(f'User set background color to {self.bgColor}')

    def handleImg(self):
        """
        处理图片（像素化）
        :return:
        """
        if self.srcImg is not None and self.colorsDict is not None:
            self.updateLog(f'User try to handle image, the dot size is {self.dotSize} pixels'
                           f' and denoise plan is '
                           f'{self.setDenoisePlanCb.itemText(self.denoisePlan)},'
                           f' and introduction plan is '
                           f'{self.setInroductionPlanCb.itemText(self.introductionPlan)}')
            tStart = time.time()
            self.dstImg = DenoiseAndPixel(self.srcImg, self.dotSize, self.dotSize, self.denoisePlan)
            self.colorsDict, self.dstImg = ColorIntroduction(self.dstImg, self.colorsDict, self.dotSize,
                                                             self.introductionPlan)
            tEnd = time.time()

            self.dstSL.setImg(self.dstImg)
            self.updateColors(1)
            self.statusBar().showMessage(f'Image processed successfully, time cost = {tEnd - tStart}s')
            self.updateLog(f'Image processed successfully, time cost = {tEnd - tStart}s')
        elif self.srcImg is None:
            self.statusBar().showMessage('Image cannot be empty!')
            self.updateLog('User try to handle image, but image is empty!')
        elif self.colorsDict is None:
            self.statusBar().showMessage('Color palette cannot be empty!')
            self.updateLog('User try to handle image, but color palette is empty!')

    def setResImg(self):
        """
        设置结果图片
        :return:
        """
        self.resImg = RoundFill(self.dstImg, self.dotSize, self.bgColor)
        self.resSL.setImg(self.resImg)

    def updateColors(self, colorsSrc):
        """
        导入/刷新颜色
        :param colorsSrc: 0:从文件导入，1:从字典导入
        :return:
        """
        if colorsSrc == 0:
            # get colors txt file from folder
            folderPath = os.path.dirname(os.path.realpath(__file__))
            filePath, fileType = QFileDialog.getOpenFileName(self, 'Choose a color palette', folderPath,
                                                             'Text File(*.txt)')
            if len(filePath) != 0:
                self.colorsPath = filePath
                self.colorsDict = self.colorPalette.setColors(self.colorsPath)
                self.updateLog(f'User import a color palette from path {self.colorsPath}')

            # self.colorsPath = './resource/colors2.txt'
            # self.colorsDict = self.colorPalette.setColors(self.colorsPath)
        else:
            self.colorsDict = self.colorPalette.setColors(self.colorsDict)
            self.updateLog('User update the color chromatogram.')

    def updateLog(self, info):
        """
        更新日志
        :param info: 新信息
        :return:
        """
        self.log = self.log + f'- {info}\n'
        self.infoTextEdit.logTextEdit.setText(self.log)

    def exportColors(self):
        """
        导出颜色
        :return:
        """
        # 导出色谱为excel
        if self.colorsDict is not None:
            # 获取当前文件夹的路径
            folderPath = os.path.dirname(os.path.realpath(__file__))
            # 保存文件
            saveFile, _ = QFileDialog.getSaveFileName(self, 'Save File', folderPath, 'File(*.xlsx *.csv)')
            fileType = saveFile.split('.')[-1]
            if len(saveFile) != 0:
                if fileType == 'xlsx':
                    # export to excel
                    exportToExcel(self.colorPalette.colorsDict, saveFile)
                elif fileType == 'csv':
                    # export To csv
                    exportToCsv(self.colorPalette.colorsDict, saveFile)
                self.statusBar().showMessage('Color palette saved!')
                self.updateLog(f'User export a color palette to path {saveFile}')
            else:
                return

    def exportImg(self):
        """
        导出图片
        :return:
        """
        if self.resImg is not None:
            # 获取当前文件夹的路径
            folderPath = os.path.dirname(os.path.realpath(__file__))
            # 保存文件
            saveFile, fileType = QFileDialog.getSaveFileName(self, 'Save File', folderPath, 'Image File(*.png *.jpg)')
            if len(saveFile) != 0:
                cv2.imwrite(saveFile, self.resImg)
            self.statusBar().showMessage('Result saved!')
        else:
            self.statusBar().showMessage('Result cannot be empty!')

    def exportLog(self):
        """
        导出日志
        :return:
        """
        folderPath = os.path.dirname(os.path.realpath(__file__))
        saveFile, fileType = QFileDialog.getSaveFileName(self, 'Save File', folderPath, 'Text File(*.txt)')
        if len(saveFile) != 0:
            with open(saveFile, 'w') as f:
                f.write(self.log)
        self.statusBar().showMessage('Log saved!')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main()
    sys.exit(app.exec())
