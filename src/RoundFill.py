import time

import cv2
import numpy as np


def RoundFill(srcImg, dotSize, bgColor):
    """
    进行圆点填充，0.017s -> 0.014s
    :param srcImg: 进行色彩归纳后的像素图
    :param dotSize: 圆点的高度
    :param bgColor: 背景色
    :return: 返回圆点填充图
    """
    height, width, _ = srcImg.shape
    # 为了形成蜂窝状
    temp = pow(3, 0.5) / 2
    dstImg = np.full((height, width, 3), bgColor, np.uint8)
    halfDotSize = int(round(dotSize / 2))
    d = int(dotSize / 4)
    ySep = int(round(temp * dotSize))
    # 记录奇数行还是偶数行
    idx = 0
    for y in range(halfDotSize, height, ySep):
        if y + dotSize * temp >= height or y - dotSize * temp < 0:
            continue
        y = int(y)
        xStart = 0 if idx % 2 == 1 else halfDotSize
        for x in range(xStart, width, dotSize):
            if x + dotSize >= width or x - dotSize < 0:
                continue

            color = srcImg[y + d][x + d]
            color = tuple(int(x) for x in color)
            cv2.circle(dstImg, (x, y), halfDotSize, color, -1)
        idx += 1
    return dstImg


if __name__ == '__main__':
    srcImg = cv2.imread('Big Sur.jpg')
    tStart = time.time()
    dstImg = RoundFill(srcImg, 10)
    tEnd = time.time()
    print(f'{tEnd - tStart}s')
    # ans = np.hstack([srcImg, dstImg])
    # cv2.imshow('s', ans)
    # cv2.waitKey(0)
