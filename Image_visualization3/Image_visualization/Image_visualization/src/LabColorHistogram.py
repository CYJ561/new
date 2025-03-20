"""
Lab颜色特征和直方图 
 对图像进行特征提取，每幅图像采用32通道进行采集
  最终获得96维向量"
cv2.imread(filename,[,flags])用于从指定的文件加载图像并返回图像的矩阵
参数说明： filename：文件路径
         flags：读取图片的方式，可选项
    ·cv2.IMREAD_COLOR(1)：始终将图像转换为 3 通道BGR彩色图像，默认方式
    ·cv2.IMREAD_GRAYSCALE(0)：始终将图像转换为单通道灰度图像
    ·cv2.IMREAD_UNCHANGED(-1)：按原样返回加载的图像（使用Alpha通道）
    ·cv2.IMREAD_ANYDEPTH(2)：在输入具有相应深度时返回16位/ 32位图像，否则将其转换为8位
    ·cv2.IMREAD_ANYCOLOR(4)：以任何可能的颜色格式读取图像
cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式
    cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
    cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
cv2.namedWindow(window_name, flag)  用于创建一个具有合适名称和大小的窗口，以在屏幕上显示图像和视频。
    window_name:将显示图像/视频的窗口的名称
    flag:表示窗口大小是自动设置还是可调整
        WINDOW_NORMAL –允许手动更改窗口大小或者WINDOW_GUI_NORMAL
        WINDOW_AUTOSIZE(Default) –自动 设置窗口大小
        WINDOW_FULLSCREEN –将窗口大小更改为全屏
cv2.imshow(winname, img) 用于在窗口中显示图像
    winname: 字符串，显示窗口的名称
    img:所显示的OpenCV图像，nparray多维数组
    cv2.imshow() 之后要用 waitKey() 函数设定图像窗口的显示时长，否则不会显示图像窗口。waitKey(0) 表示窗口显示时长为无限。
"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


class LabColorHistogram:

    def __init__(self, path):
        self.path = path
        # self.Lab()

    def Lab(self):  # 图像特征处理（96维）

        images_processed = os.listdir(self.path)  # 返回指定路径下的文件夹列表。
        # print(images_processed)  # 图片名称

        Lab = np.empty((1, 96), dtype=np.float32)  # 创建一个1*96的空数组
        Lab = np.delete(Lab, 0, axis=0)  # del第一行
        for photo_name in images_processed:
            # 图片路径
            photo_file = self.path + photo_name
            # 通过指定文件路径读取图像，返回图像矩阵
            photo_BGR = cv2.imread(photo_file, cv2.IMREAD_COLOR)  # 将图像转换为3通道BGR彩色图像
            photo_RGB = cv2.cvtColor(photo_BGR, cv2.COLOR_BGR2LAB)  # 将BGR格式转换成RGB格式
            # cv2.namedWindow("input", cv2.WINDOW_GUI_NORMAL)  # 创建一个具有合适名称和大小的窗口
            # cv2.imshow("input", photo_RGB)  # 用于在窗口中显示图像
            # cv2.waitKey(0)   # 设定图像窗口的显示时长

            # 32维向量，共32*3=96
            b_vector = cv2.calcHist([photo_BGR], [0], None, [32], [0, 255]).transpose()  # transpose()可对换数组
            g_vector = cv2.calcHist([photo_BGR], [1], None, [32], [0, 255]).transpose()
            r_vector = cv2.calcHist([photo_BGR], [2], None, [32], [0, 255]).transpose()

            # plt.plot(b_vector.transpose(), label='B', color='blue')
            # plt.plot(g_vector.transpose(), label='G', color='green')
            # plt.plot(r_vector.transpose(), label='R', color='red')
            # plt.legend(loc='best')
            # plt.xlim([0, 33])
            # plt.show()
            # cv2.waitKey(0)
            Lab_vector = np.hstack((b_vector, g_vector, r_vector))  # 将三个向量合并到同一行

            # print(b_vector)
            # 直接插入
            Lab = np.append(Lab, Lab_vector, axis=0)  # axis = 0表示往下排
        print("Lab初始图像集维度为", Lab.shape)
        return Lab


# a = LabColorHistogram(path="F:\\PythonPro\\Image_visualization\\Destination\\")
# b = a.Lab()
# print(b)
