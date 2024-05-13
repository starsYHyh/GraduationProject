import time

import numpy as np
import cv2

if __name__ == '__main__':
    srcImg = cv2.imread('resource/t2.jpg')
    ts = time.time()
    height, width = srcImg.shape[:2]
    # 缩小成改尺寸的大小
    tempImg = cv2.resize(srcImg, (int(round(width / 10)), int(round(height / 10))), interpolation=cv2.INTER_LINEAR)
    te = time.time()
    print(f'{te - ts}s')
