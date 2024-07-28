# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np

img = cv2.imread('1.jpg')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转成RGB 方便后面显示

# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Roberts算子
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


lines = cv2.HoughLinesP(Roberts, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# 创建一个空白图像用于绘制网状物体
mesh = np.zeros_like(img_RGB)

# 绘制检测到的直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(mesh, (x1, y1), (x2, y2), (255, 255, 255), 2)

# 进行形态学操作，填充闭合区域
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(mesh, cv2.MORPH_CLOSE, kernel)

closing_gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

# 阈值处理，得到二值图像
_, threshold = cv2.threshold(closing_gray, 127, 255, cv2.THRESH_BINARY)
# 寻找轮廓
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制检测到的网状物体轮廓
cv2.drawContours(img_RGB, contours, -1, (0, 255, 0), 2)
# # 自适应阈值化
# binary = cv2.adaptiveThreshold(
#         Roberts, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 1001, 1)
# #ret, binary = cv2.threshold(Roberts, 0, 255, cv2.THRESH_BINARY)
# # 先进行膨胀操作，填充孔洞
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# dilated = cv2.dilate(binary, kernel, iterations=1)
#
# # 再进行腐蚀操作，恢复原始形状
# eroded = cv2.erode(dilated, kernel, iterations=1)
# contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     area = cv2.contourArea(contour)
#     print(area)
#     # 根据预期孔洞面积范围进行筛选
#     if area > 10 and area < 10000:
#         # 绘制检测到的孔洞的边界框或其他操作
#         print("1")
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(img_RGB, (x, y), (x + w, y + h), (0, 255, 0), 2)

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.imshow(img_RGB), plt.title('RGB'), plt.axis('off')
#plt.imshow(Roberts), plt.title('Roberts算子'), plt.axis('off')
#plt.imshow(Roberts, cmap=plt.cm.gray), plt.title('Roberts算子'), plt.axis('off')
plt.show()
