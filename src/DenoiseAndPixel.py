import time

import numpy as np
import cv2


def NLMDenoising(srcImg):
    """
    用于降噪，采用NLM方法
    :param srcImg: 原图像
    :return: 结果
    """
    dstImg = cv2.fastNlMeansDenoisingColored(srcImg, None, 10, 10, 7, 21)
    return dstImg


def BilateralDenoising(srcImg):
    dstImg = cv2.bilateralFilter(srcImg, 7, 75, 75)
    return dstImg


def meanBlur(srcImg):
    """
    均值滤波
    :param srcImg: 原图像
    :return: 结果
    """
    dstImg = cv2.blur(srcImg, (7, 7))
    return dstImg


def Pixelate(srcImg, h=10, w=10):
    """
    将图像进行像素化处理
    :param srcImg: 原图像
    :param h: 每个像素格子的高度
    :param w: 每个像素格子的宽度
    :return: 返回结果图像
    """
    height, width = srcImg.shape[:2]
    # 缩小成改尺寸的大小
    tempImg = cv2.resize(srcImg, (int(round(width / w)), int(round(height / h))), interpolation=cv2.INTER_LINEAR)
    # 输出图像
    dstImg = cv2.resize(tempImg, (width, height), interpolation=cv2.INTER_NEAREST)
    return dstImg


def NEAR_Pixelate(srcImg, h=10, w=10):
    """
    将图像进行像素化处理
    :param srcImg: 原图像
    :param h: 每个像素格子的高度
    :param w: 每个像素格子的宽度
    :return: 返回结果图像
    """
    height, width = srcImg.shape[:2]
    # 缩小成改尺寸的大小
    tempImg = cv2.resize(srcImg, (int(round(width / w)), int(round(height / h))), interpolation=cv2.INTER_NEAREST)
    # 输出图像
    dstImg = cv2.resize(tempImg, (width, height), interpolation=cv2.INTER_NEAREST)
    return dstImg


def LINEAR_Pixelate(srcImg, h=10, w=10):
    """
    将图像进行像素化处理
    :param srcImg: 原图像
    :param h: 每个像素格子的高度
    :param w: 每个像素格子的宽度
    :return: 返回结果图像
    """
    height, width = srcImg.shape[:2]
    # 缩小成改尺寸的大小
    tempImg = cv2.resize(srcImg, (int(round(width / w)), int(round(height / h))), interpolation=cv2.INTER_LINEAR)
    # 输出图像
    dstImg = cv2.resize(tempImg, (width, height), interpolation=cv2.INTER_NEAREST)
    return dstImg


def CUBIC_Pixelate(srcImg, h=10, w=10):
    """
    将图像进行像素化处理
    :param srcImg: 原图像
    :param h: 每个像素格子的高度
    :param w: 每个像素格子的宽度
    :return: 返回结果图像
    """
    height, width = srcImg.shape[:2]
    # 缩小成改尺寸的大小
    tempImg = cv2.resize(srcImg, (int(round(width / w)), int(round(height / h))), interpolation=cv2.INTER_CUBIC)
    # 输出图像
    dstImg = cv2.resize(tempImg, (width, height), interpolation=cv2.INTER_NEAREST)
    return dstImg


def DenoiseAndPixel(srcImg, h, w, denoisePlan):
    """
    对图像进行降噪和像素化处理
    :param srcImg: 原图像
    :param h: 图像高度
    :param w: 图像宽度
    :param denoisePlan: 降噪方案
    :return:
    """
    if denoisePlan == 1:
        tempImg = meanBlur(srcImg)
    elif denoisePlan == 2:
        tempImg = BilateralDenoising(srcImg)
    elif denoisePlan == 3:
        tempImg = NLMDenoising(srcImg)
    else:
        if srcImg.shape[0] * srcImg.shape[1] > 4000 * 4000:
            tempImg = meanBlur(srcImg)
        elif 2000 * 2000 < srcImg.shape[0] * srcImg.shape[1] <= 4000 * 4000:
            tempImg = BilateralDenoising(srcImg)
        else:
            tempImg = NLMDenoising(srcImg)
    dstImg = Pixelate(tempImg, h=h, w=w)
    return dstImg


if __name__ == '__main__':
    testImgs = ['resource/t1.jpg', 'resource/t2.jpg', 'resource/t3.jpg', 'resource/t4.jpg']
    testTimes = []
    for i in range(len(testImgs)):
        testTimes.append([])
        srcImg = cv2.imread(testImgs[i])
        srcImg = cv2.bilateralFilter(srcImg, 7, 75, 75)

        ts = time.time()
        dstImg1 = NEAR_Pixelate(srcImg, 10)
        te = time.time()
        cv2.imwrite(f'pixel/t{i}1.jpg', dstImg1)
        testTimes[i].append(te - ts)

        ts = time.time()
        dstImg2 = LINEAR_Pixelate(srcImg, 10)
        te = time.time()
        cv2.imwrite(f'pixel/t{i}2.jpg', dstImg2)
        testTimes[i].append(te - ts)

        ts = time.time()
        dstImg3 = CUBIC_Pixelate(srcImg, 10)
        te = time.time()
        cv2.imwrite(f'pixel/t{i}3.jpg', dstImg3)
        testTimes[i].append(te - ts)

    # dstImgs = [['resource/t1.jpg', 'denoise/t01.jpg', 'denoise/t02.jpg', 'denoise/t03.jpg'],
    #            ['resource/t2.jpg', 'denoise/t11.jpg', 'denoise/t12.jpg', 'denoise/t13.jpg'],
    #            ['resource/t3.jpg', 'denoise/t21.jpg', 'denoise/t22.jpg', 'denoise/t23.jpg'],
    #            ['resource/t4.jpg', 'denoise/t31.jpg', 'denoise/t32.jpg', 'denoise/t33.jpg']]
    # for i in range(len(dstImgs)):
    #     ansImg = np.hstack((cv2.imread(dstImgs[i][0]), cv2.imread(dstImgs[i][1]), cv2.imread(dstImgs[i][2]), cv2.imread(dstImgs[i][3])))
    #     cv2.imwrite(f'denoise/stack_t{i}.jpg', ansImg)


    print()
