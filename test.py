import cv2 as cv
import os
import numpy as np

img_dir = ".\img"
img_list = os.listdir(img_dir)
img_list = list(map(lambda x: os.path.join(img_dir, x), img_list))
print(img_list)

# im = cv.imread(img_list[2])
im = cv.imread(".//img//0.png")
# im = cv.resize(im, (500,1000))
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 椎骨rect信息列表，y值从小到大
rect_list = []
for index in range(len(contours)):
    contour = contours[index]
    rect = cv.minAreaRect(contour)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    rect_list.append(rect)
rect_list.sort(key=lambda item: item[0][1])  # 按照y值从小到大排序

# 椎骨的端点信息
vertebrae_vertex = []  # 已经按照y轴从小打大排序
for index in range(len(rect_list)):
    point = (int(rect_list[index][0][0]), int(rect_list[index][0][1]))
    point2 = (int(rect_list[index][0][0])+50, int(rect_list[index][0][1]))
    im = cv.putText(im, str(index), point, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    im = cv.putText(im, str(rect_list[index][2]), point2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    print(index, "degree: ", rect_list[index][2])

    # 保存椎骨定点信息
    box = cv.boxPoints(rect_list[index])
    box = np.int0(box)
    list_box = list(box)
    list_box.sort(key=lambda item: item[1])
    vertebrae_vertex.append(list_box)
    im = cv.drawContours(im, [box], 0, (0, 0, 255), 1)

cv.namedWindow("out", cv.WINDOW_NORMAL)
cv.imshow("out", im)
cv.waitKey(0)
