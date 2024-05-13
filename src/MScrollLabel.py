import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QVBoxLayout, QScrollArea, QWidget, QLabel


class MLabel(QLabel):
    def __init__(self):
        super().__init__()

    def paintEvt(self, event):
        super().paintEvent(event)


class MScrollLabel(QWidget):
    """
    这是一个用于展示图片的带滚动条的区域
    """

    def __init__(self, h, w):
        """
        构造器
        :param h: 展示区域的高度
        :param w: 展示区域的宽度
        """
        super().__init__()

        self.scroll = QScrollArea(self)
        self.label = MLabel()
        self.label.enterEvent = self.enterEvent
        self.label.leaveEvent = self.leaveEvent
        self.scroll.setWidget(self.label)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('QScrollArea{'
                           'background-color: #fff;'
                           'border: 1px solid grey;'
                           'border-radius: 5px;'
                           '}')

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll)

        # scrollWidth = self.scroll.verticalScrollBar().width()
        # scrollHeight = self.scroll.horizontalScrollBar().height()
        self.scrollWidth = 20
        self.scrollHeight = 20

        self.scroll.viewport().resize(w, h)
        self.resize(w + 2 * self.scrollWidth, h + 2 * self.scrollHeight)

    def setImg(self, srcImg):
        """
        展示图片
        :param srcImg: opencv格式图像
        :return:
        """
        srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)
        height, width, channel = srcImg.shape
        bytesPerLine = channel * width
        qImg = QImage(srcImg.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)
        width = pixmap.width()
        height = pixmap.height()

        self.label.resize(width, height)

    def resizeEvent(self, event):
        # width = self.width() - 2 * self.scrollWidth
        # height = self.height() - 2 * self.scrollHeight
        self.scroll.setWidget(self.label)

    # 当鼠标进入图像区域时，会把指针形状变成十字形
    def enterEvent(self, event):
        self.label.setCursor(Qt.CursorShape.CrossCursor)

    def leaveEvent(self, event):
        self.label.setCursor(Qt.CursorShape.ArrowCursor)
