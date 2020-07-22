import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
from math import sqrt
from math import atan
import math

img_dir = ".\saggital"
img_list = os.listdir(img_dir)
print(img_list)
output = ".\\test2"

yellow = (0, 255, 255)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
font = cv.FONT_HERSHEY_COMPLEX
name_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10',
             'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
id_t2 = 8
id_t5 = 11
id_t10 = 16
id_l2 = 20

print(len(name_list))
for item in tqdm(img_list):
    file = os.path.join(img_dir, item)
    output_file = os.path.join(output, item)

    im = cv.imread(file)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 100, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 椎骨rect信息列表，y值从小到大
    rect_list = []
    final_contour_info = []
    final_contour = []
    for index in range(len(contours)):
        contour = contours[index]
        area = cv.contourArea(contour)
        if area > 2000:
            # print(area)
            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)
            contour_info = {'center': center, 'contour': contour}
            final_contour_info.append(contour_info)
            final_contour.append(contour)
    cv.drawContours(im, final_contour, -1, green, 5)
    # print(len(final_contour))

    sort_contours = final_contour_info.copy()
    sort_contours.sort(key=lambda x: x['center'][1])
    print(len(sort_contours))
    for id in range(len(sort_contours)):
        center = (sort_contours[id]['center'][0] - 50, sort_contours[id]['center'][1])
        im = cv.putText(im, str(name_list[id]), center, cv.FONT_HERSHEY_SIMPLEX, 2, blue, 3)

    cal_con = [sort_contours[id_t2], sort_contours[id_t5], sort_contours[id_t10], sort_contours[id_l2]]
    cal_vertix = []
    for item in cal_con:
        cnt = item['contour']
        approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)
        n = approx.ravel()  # 变为一维数组
        vertexs = []
        for i in range(8):
            if i % 2 == 0:
                point = (n[i], n[i + 1])
                vertexs.append(point)
        if len(vertexs) == 4:
            cal_vertix.append(vertexs)

    # 需要计算的顶点调整位置
    for item in cal_vertix:
        item.sort(key=lambda x: x[1])
        for id in range(len(item)):
            cv.putText(im, str(id), item[id], font, 2, yellow, 3)
    slope_up = []
    slope_down = []
    count = 0
    for item in cal_vertix:
        up_slope = (item[0][1] - item[1][1]) / (item[0][0] - item[1][0])
        down_slope = (item[2][1] - item[3][1]) / (item[2][0] - item[3][0])
        slope_up.append(up_slope)
        slope_down.append(down_slope)

        if count == 0:
            cv.line(im, item[0], item[1], red, 20)
        if count == 1:
            cv.line(im, item[2], item[3], red, 20)
        if count == 2:
            cv.line(im, item[0], item[1], blue, 20)
        if count == 3:
            cv.line(im, item[2], item[3], blue, 20)
        count += 1
    k1 = slope_up[0]
    k2 = slope_down[1]
    k3 = slope_up[2]
    k4 = slope_down[3]
    cobb1 = math.degrees(atan(abs((k2 - k1) / (1 + k1 * k2))))
    cobb2 = math.degrees(atan(abs((k3 - k4) / (1 + k3 * k4))))
    im = cv.putText(im, str(round(cobb1, 2)), (500, 500), cv.FONT_HERSHEY_SIMPLEX, 4,
                    red, 5)
    im = cv.putText(im, str(round(cobb2, 2)), (500, 1500), cv.FONT_HERSHEY_SIMPLEX, 4,
                    blue, 5)
    cv.imwrite(output_file, im)
