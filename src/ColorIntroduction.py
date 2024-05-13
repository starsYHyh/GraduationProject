import cv2
import numpy as np
import numba as nb
import time
from DenoiseAndPixel import DenoiseAndPixel
from Helper import mcie2000M

kl = 2


@nb.jit(nopython=True)
def Introduction_1(srcImg, labColorsMap, rgbColorsMap, dstImg, h, w):
    """
    色彩归纳，适用于大尺寸图片，速度稳定
    :param srcImg: 原图像(Lab)
    :param labColorsMap: 色谱(opencv Lab)，用于比较差异
    :param rgbColorsMap: 色谱(rgb)，为了避免rgb与LAB相互转换时的造成的误差，使用原色谱的rgb值进行赋值操作
    :param dstImg: 目标图像
    :param h: 图像的高度
    :param w: 图像的高度
    :return: 返回结果色谱每一个颜色所有出现的位置
    """
    height, width, _ = srcImg.shape
    # 为了使用加速，需要进行此操作告诉numba编译器
    tempColorsInfo = [[[-1, -1]]]
    for i in range(len(labColorsMap) - 1):
        tempColorsInfo.append([[-1, -1]])
    labColorsMap = labColorsMap.astype(np.int32)
    srcImg = srcImg.astype(np.int32)

    for y in range(0, height, h):
        for x in range(0, width, w):
            # Lab颜色空间上的欧氏距离（的平方）
            maxdLab = 3 * (255 ** 2 * (kl ** 0.5)) + 100
            idx = 0
            # 进行算法的优化
            for i in range(len(labColorsMap)):
                cur = kl * (labColorsMap[i][0] - srcImg[y][x][0]) ** 2 + (
                        labColorsMap[i][1] - srcImg[y][x][1]) ** 2 + (
                              labColorsMap[i][2] - srcImg[y][x][2]) ** 2
                if cur < maxdLab:
                    maxdLab = cur
                    idx = i
            tempColorsInfo[idx].append([y, x])

            # 赋值
            for a in range(y, min(y + h, height)):
                for b in range(x, min(x + w, width)):
                    # rgb to bgr
                    dstImg[a][b][0] = rgbColorsMap[idx][2]
                    dstImg[a][b][1] = rgbColorsMap[idx][1]
                    dstImg[a][b][2] = rgbColorsMap[idx][0]

    for i in range(len(tempColorsInfo)):
        tempColorsInfo[i] = tempColorsInfo[i][1:]
    return tempColorsInfo


def Introduction_2(srcImg, labColorsMap, rgbColorsMap, dstImg, h, w):
    """
    色彩归纳，适用于中小尺寸图片，极致速度
    :param srcImg: 原图像(Lab)
    :param labColorsMap: 色谱(Lab)，用于比较差异
    :param rgbColorsMap: 色谱(rgb)，为了避免rgb与LAB相互转换时的造成的误差，使用原色谱的rgb值进行赋值操作
    :param dstImg: 目标图像
    :param h: 图像的高度
    :param w: 图像的高度
    :return: 返回结果色谱每一个颜色所有出现的位置
    """
    height, width, _ = srcImg.shape
    tempColorsInfo = [[] for i in range(len(labColorsMap))]
    labColorsMap = labColorsMap.astype(np.int32)
    srcImg = srcImg.astype(np.int32)

    for y in range(0, height, h):
        for x in range(0, width, w):
            # 进行算法的优化
            # 优化点，使用numpy的广播机制，1.06s -> 0.06s
            dLab = kl * (labColorsMap[:, 0] - srcImg[y][x][0]) ** 2 + (labColorsMap[:, 1] - srcImg[y][x][1]) ** 2 + (
                    labColorsMap[:, 2] - srcImg[y][x][2]) ** 2
            idx = np.argmin(dLab)
            tempColorsInfo[idx].append([y, x])
            dstImg[y:y + h, x:x + w] = rgbColorsMap[idx][[2, 1, 0]]
    return tempColorsInfo


def Introduction_3(srcImg, labColorsMap, rgbColorsMap, dstImg, h, w):
    """
    色彩归纳，准确度优先，适用于小尺寸图片
    :param srcImg: 原图像(Lab)
    :param labColorsMap: 色谱(Lab)，用于比较差异
    :param rgbColorsMap: 色谱(rgb)，为了避免rgb与LAB相互转换时的造成的误差，使用原色谱的rgb值进行赋值操作
    :param dstImg: 目标图像
    :param h: 图像的高度
    :param w: 图像的高度
    :return: 返回结果色谱每一个颜色所有出现的位置
    """
    height, width, _ = srcImg.shape
    tempColorsInfo = [[] for i in range(len(labColorsMap))]

    lab1 = srcImg.astype(np.float64)
    lab1[..., 0] = (lab1[..., 0] * 100.0 / 255.0)
    lab1[..., 1:] = (lab1[..., 1:] - 128.0) * 255.0 / 255.0

    lab2 = np.array([labColorsMap]).astype(np.float64)[0]
    lab2[..., 0] = (lab2[..., 0] * 100.0 / 255.0)
    lab2[..., 1:] = (lab2[..., 1:] - 128.0) * 255.0 / 255.0

    for y in range(0, height, h):
        for x in range(0, width, w):
            # 优化点，使用CIEDE2000算法和argmin函数
            curs = mcie2000M(lab1[y][x], lab2)
            idx = np.argmin(curs)
            tempColorsInfo[idx].append([y, x])

            # 赋值
            # 优化点，使用numpy的切片操作
            dstImg[y:y + h, x:x + w] = rgbColorsMap[idx][[2, 1, 0]]
    return tempColorsInfo


def Introduction_4(srcImg, labColorsMap, rgbColorsMap, dstImg, h, w):
    """
    更极致速度，be king写法
    :param srcImg: 原图像(Lab)
    :param labColorsMap: 色谱(Lab)，用于比较差异
    :param rgbColorsMap: 色谱(rgb)，为了避免rgb与LAB相互转换时的造成的误差，使用原色谱的rgb值进行赋值操作
    :param dstImg: 目标图像
    :param h: 图像的高度
    :param w: 图像的高度
    :return: 返回结果色谱每一个颜色所有出现的位置
    """
    height, width, _ = srcImg.shape
    tempColorsInfo = [[] for i in range(len(labColorsMap))]
    labColorsMap = labColorsMap.astype(np.int32)
    srcImg = srcImg.astype(np.int32)

    # 进一步优化
    # 优化点，使用meshgrid()和nindex来避免双重循环，0.06s -> 0.02s
    x_coords = np.arange(0, width, w)
    y_coords = np.arange(0, height, h)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='xy')

    dLab = (labColorsMap[:, 0, np.newaxis, np.newaxis] - srcImg[yy, xx, 0]) ** 2 + \
           kl * (labColorsMap[:, 1, np.newaxis, np.newaxis] - srcImg[yy, xx, 1]) ** 2 + \
           kl * (labColorsMap[:, 2, np.newaxis, np.newaxis] - srcImg[yy, xx, 2]) ** 2

    idx = np.argmin(dLab, axis=0)
    for i, j in np.ndindex(idx.shape):
        y = y_coords[i]
        x = x_coords[j]
        tempColorsInfo[idx[i, j]].append([y, x])
        dstImg[y: y + h, x:x + w] = rgbColorsMap[idx[i, j], ::-1]
    return tempColorsInfo


def Introduction_5(srcImg, labColorsMap, rgbColorsMap, dstImg, h, w):
    """
    色彩归纳
    :param srcImg: 原图像(Lab)
    :param labColorsMap: 色谱(Lab)，用于比较差异
    :param rgbColorsMap: 色谱(rgb)，为了避免rgb与LAB相互转换时的造成的误差，使用原色谱的rgb值进行赋值操作
    :param dstImg: 目标图像
    :param h: 图像的高度
    :param w: 图像的高度
    :return: 返回结果色谱每一个颜色所有出现的位置
    """
    height, width, _ = srcImg.shape
    # 为了使用加速，需要进行此操作告诉numba编译器
    tempColorsInfo = [[[-1, -1]]]
    tempLabColorsMap = labColorsMap.astype(np.int32)
    for i in range(len(tempLabColorsMap) - 1):
        tempColorsInfo.append([[-1, -1]])

    tempSrcImg = srcImg.astype(np.int32)

    for y in range(0, height, h):
        for x in range(0, width, w):
            # Lab颜色空间上的欧氏距离（的平方）
            index = 0
            curColor = tempSrcImg[y][x]
            Si = curColor[0] + curColor[1] + curColor[2]

            # 进行算法的优化
            # 先找出与当前颜色范数（的平方）最接近的颜色
            mNorm = (curColor[0] ** 2 + kl * (curColor[1] ** 2) + kl * (curColor[1] ** 2))
            l = 0
            r = len(tempLabColorsMap)

            count = 0
            while l <= r:
                m = (l + r) // 2
                curNorm = (tempLabColorsMap[m][0] ** 2 + kl * (tempLabColorsMap[m][1] ** 2) + kl * (tempLabColorsMap[m][1] ** 2))
                if curNorm < mNorm:
                    l = m + 1
                elif curNorm > mNorm:
                    r = m - 1
                else:
                    index = m
                    break
                count += 1
            if l > r:
                lNorm = (tempLabColorsMap[l][0] ** 2 + kl * (tempLabColorsMap[l][1] ** 2) + kl * (tempLabColorsMap[l][1] ** 2))
                rNorm = (tempLabColorsMap[r][0] ** 2 + kl * (tempLabColorsMap[r][1] ** 2) + kl * (tempLabColorsMap[r][1] ** 2))
                index = l if abs(lNorm - mNorm) < abs(rNorm - mNorm) else r
            pos = index + 1

            # 初始SED
            D0 = (curColor[0] - tempLabColorsMap[index][0]) ** 2 + \
                 kl * ((curColor[1] - tempLabColorsMap[index][1]) ** 2) + \
                 kl * ((curColor[2] - tempLabColorsMap[index][2]) ** 2)
            Ci = (curColor[0] ** 2 + kl * (curColor[1] ** 2) + kl * (curColor[2] ** 2)) ** 0.5
            # 从当前索引出发，向前找
            lNormBound = Ci - D0 ** 0.5
            rNormBound = Ci + D0 ** 0.5
            lSumBound = Si - (3 * D0) ** 0.5
            rSumBound = Si + (3 * D0) ** 0.5

            minD = D0
            cur = index
            while cur >= 0:
                Ck = (tempLabColorsMap[cur][0] ** 2 + kl * (tempLabColorsMap[cur][1] ** 2) + kl * (tempLabColorsMap[cur][2] ** 2)) ** 0.5
                Sk = tempLabColorsMap[cur][0] + tempLabColorsMap[cur][1] + tempLabColorsMap[cur][2]
                if Ck <= lNormBound or Sk <= lSumBound:
                    break
                curD = (curColor[0] - tempLabColorsMap[cur][0]) ** 2 + \
                       kl * ((curColor[1] - tempLabColorsMap[cur][1]) ** 2) + \
                       kl * ((curColor[2] - tempLabColorsMap[cur][2]) ** 2)
                if minD > curD:
                    minD = curD
                    index = cur
                cur -= 1
                count += 1
            cur = pos + 1
            while cur < len(tempLabColorsMap):
                Ck = (tempLabColorsMap[cur][0] ** 2 + kl * (tempLabColorsMap[cur][1] ** 2) + kl * (tempLabColorsMap[cur][2] ** 2)) ** 0.5
                Sk = tempLabColorsMap[cur][0] + tempLabColorsMap[cur][1] + tempLabColorsMap[cur][2]
                if Ck >= rNormBound or Sk >= rSumBound:
                    break
                curD = (curColor[0] - tempLabColorsMap[cur][0]) ** 2 + \
                       kl * ((curColor[1] - tempLabColorsMap[cur][1]) ** 2) + \
                       kl * ((curColor[2] - tempLabColorsMap[cur][2]) ** 2)
                if minD > curD:
                    minD = curD
                    index = cur
                cur += 1
                count += 1

            # # 暴力查找
            # maxdLab = 3 * (255 ** 2 * (k ** 0.5)) + 100
            # for i in range(len(labColorsMap)):
            #     cur = (int(labColorsMap[i][0]) - int(srcImg[y][x][0])) ** 2 + k * (
            #             int(labColorsMap[i][1]) - int(srcImg[y][x][1])) ** 2 + k * (
            #                   int(labColorsMap[i][2]) - int(srcImg[y][x][2])) ** 2
            #     if cur < maxdLab:
            #         maxdLab = cur
            #         index = i

            # dstImg[y][x] = colorsMap[idx]
            tempColorsInfo[index].append([y, x])

            # 赋值
            for a in range(y, min(y + h, height)):
                for b in range(x, min(x + w, width)):
                    # b g r
                    dstImg[a][b][0] = rgbColorsMap[index][2]
                    dstImg[a][b][1] = rgbColorsMap[index][1]
                    dstImg[a][b][2] = rgbColorsMap[index][0]


    for i in range(len(tempColorsInfo)):
        tempColorsInfo[i] = tempColorsInfo[i][1:]
    return tempColorsInfo


def getColors(colorsDict):
    # 从字典中获取颜色，并转为lab和rgb
    colorsStr = list(colorsDict.keys())
    colors = np.array([[list(map(int, i.strip().split(','))) for i in colorsStr]], dtype=np.uint8)
    labColors, rgbColors = cv2.cvtColor(colors, cv2.COLOR_RGB2LAB)[0], colors[0]
    return labColors, rgbColors


def ColorIntroduction(srcImg, colorsDict, dotSize, introductionPlan):
    """
    颜色归纳
    :param srcImg: 已经像素化处理过的图片
    :param colorsDict: 色谱的字典
    :param dotSize: 像素块的大小
    :param introductionPlan: 归纳方案
    :return: 颜色信息和结果图像
    """
    labColorsMap, rgbColorsMap = getColors(colorsDict)
    labColorsMap = np.array([labColorsMap])[0]
    rgbColorsMap = np.array([rgbColorsMap])[0]

    dstImg = np.zeros(srcImg.shape, dtype=srcImg.dtype)
    LabImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2LAB)

    tStart = time.time()
    if introductionPlan == 2:
        colorsInfo = Introduction_3(srcImg=LabImg, labColorsMap=labColorsMap, rgbColorsMap=rgbColorsMap, dstImg=dstImg,
                                    h=dotSize, w=dotSize)
    elif introductionPlan == 1:
        if srcImg.shape[0] * srcImg.shape[1] > 2000 * 2000:
            colorsInfo = Introduction_1(srcImg=LabImg, labColorsMap=labColorsMap, rgbColorsMap=rgbColorsMap,
                                        dstImg=dstImg, h=dotSize, w=dotSize)
        else:
            colorsInfo = Introduction_4(srcImg=LabImg, labColorsMap=labColorsMap, rgbColorsMap=rgbColorsMap,
                                        dstImg=dstImg, h=dotSize, w=dotSize)
    else:
        if srcImg.shape[0] * srcImg.shape[1] > 2000 * 2000:
            colorsInfo = Introduction_1(srcImg=LabImg, labColorsMap=labColorsMap, rgbColorsMap=rgbColorsMap,
                                        dstImg=dstImg, h=dotSize, w=dotSize)
        elif 800 * 800 < srcImg.shape[0] * srcImg.shape[1] <= 2000 * 2000:
            colorsInfo = Introduction_2(srcImg=LabImg, labColorsMap=labColorsMap, rgbColorsMap=rgbColorsMap,
                                        dstImg=dstImg, h=dotSize, w=dotSize)
        else:
            colorsInfo = Introduction_3(srcImg=LabImg, labColorsMap=labColorsMap, rgbColorsMap=rgbColorsMap,
                                        dstImg=dstImg, h=dotSize, w=dotSize)
    tEnd = time.time()
    print(f'{tEnd - tStart}s')

    # 使用字典存储可以有效加快查询速度
    colorsDict = {f'{rgbColorsMap[i][0]}, {rgbColorsMap[i][1]}, {rgbColorsMap[i][2]}': colorsInfo[i] for i in
                  range(len(colorsInfo))}
    return colorsDict, dstImg


# if __name__ == '__main__':
#     # # srcImg = cv2.imread('resource/t1.jpg')
#     # colorsPath = 'resource/colors2.txt'
#     # # dst = DenoiseAndPixel(srcImg, 10, 10)
#     # # x, y, dst = ColorIntroduction(dst, colorsPath, 10)
#     # # cv2.imshow('ans', dst)
#     # # cv2.waitKey(0)
#     # testImgs = ['Introduction/t0.jpg', 'Introduction/t1.jpg', 'Introduction/t2.jpg', 'Introduction/t3.jpg']
#     # for testImg in testImgs:
#     #     dst = cv2.imread(testImg)
#     #     x, y, dst = ColorIntroduction(dst, colorsPath, 10)
#     #     print()
#
#     dstImgs = [['Introduction/t0.jpg', 'Introduction/t01.jpg', 'Introduction/t02.jpg'],
#                ['Introduction/t1.jpg', 'Introduction/t11.jpg', 'Introduction/t12.jpg'],
#                ['Introduction/t2.jpg', 'Introduction/t21.jpg', 'Introduction/t22.jpg'],
#                ['Introduction/t3.jpg', 'Introduction/t31.jpg', 'Introduction/t32.jpg']]
#     for i in range(len(dstImgs)):
#         ansImg = np.hstack((cv2.imread(dstImgs[i][0]), cv2.imread(dstImgs[i][1]), cv2.imread(dstImgs[i][2])))
#         cv2.imwrite(f'Introduction/stack_t{i}.jpg', ansImg)
#
#
#     print()

if __name__ == '__main__':
    srcImg = cv2.imread('Introduction/t4.jpg')
    x, y, ansImg = ColorIntroduction(srcImg, 'resource/colors2.txt', 10)
    # cv2.imshow('ans', ansImg)
    print('ok')
