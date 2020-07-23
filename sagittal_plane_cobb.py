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
name_list = ['L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4', 'T3', 'T2', 'T1',
             'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']
id_t2 = 8
id_t5 = 11
id_t10 = 16
id_l2 = 20

for item in tqdm(img_list):
    file = os.path.join(img_dir, item)
    output_file = os.path.join(output, item)

    im = cv.imread(file)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 100, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 筛选出真正的椎骨
    # 面积过小的不要，最后一个不要
    rect_list = []
    final_contour_info = []
    final_contour = []
    for index in range(len(contours)):
        contour = contours[index]
        area = cv.contourArea(contour)
        if 2000 < area < 40000:
            # print(area)
            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)
            contour_info = {'center': center, 'contour': contour}
            final_contour_info.append(contour_info)

    # 倒序所有轮廓，并去除最下面的轮廓
    sort_contours = final_contour_info.copy()
    sort_contours.sort(key=lambda x: x['center'][1], reverse=True)
    # sort_contours = sort_contours[1:]
    # print(len(sort_contours))

    # 将所有脊柱块命名
    new_vertebra = []
    for id in range(len(sort_contours)):
        # 画出轮廓信息和名字
        cv.drawContours(im, sort_contours[id]['contour'], -1, green, 5)
        center = (sort_contours[id]['center'][0] - 50, sort_contours[id]['center'][1])
        im = cv.putText(im, str(name_list[id]), center, cv.FONT_HERSHEY_SIMPLEX, 2, blue, 3)

        # 计算脊柱块的四个点
        cnt = sort_contours[id]['contour']
        approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)
        n = approx.ravel()  # 变为一维数组
        points = []
        # 都只存四个点
        for i in range(8):
            if i % 2 == 0:
                point = (n[i], n[i + 1])
                points.append(point)
        points.sort(key=lambda x: x[1])  # 四个点按照上下排序
        vertebra = {'name': name_list[id], 'center': sort_contours[id]['center'], 'points': points,
                    'contour': sort_contours[id]['contour']}
        new_vertebra.append(vertebra)

    calculate_vertebra = []
    for item in new_vertebra:
        if item['name'] == 'T2' or item['name'] == 'T5' or item['name'] == 'T10' or item['name'] == 'L2':
            points = item['points']
            up_slope = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
            down_slope = (points[2][1] - points[3][1]) / (points[2][0] - points[3][0])
            cal = {'name': item['name'], 'up_slope': up_slope, 'down_slope': down_slope, 'points':points}
            calculate_vertebra.append(cal)
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0
    putpoint0 = (0, 0)
    putpoint1 = (0, 0)
    for item in calculate_vertebra:
        if item['name'] == 'T2':
            k1 = item['up_slope']
            cv.line(im, item['points'][0], item['points'][1], red, 20)
            putpoint0 = ( item['points'][0][0]-600, item['points'][0][1])
        if item['name'] == 'T5':
            k2 = item['down_slope']
            cv.line(im, item['points'][2], item['points'][3], red, 20)
        if item['name'] == 'T10':
            k3 = item['up_slope']
            cv.line(im, item['points'][0], item['points'][1], green, 20)
            putpoint1 = (item['points'][0][0] - 600, item['points'][0][1])
        if item['name'] == 'L2':
            k4 = item['down_slope']
            cv.line(im, item['points'][2], item['points'][3], green, 20)

    cobb1 = math.degrees(atan(abs((k2 - k1) / (1 + k1 * k2))))
    cobb2 = math.degrees(atan(abs((k3 - k4) / (1 + k3 * k4))))
    im = cv.putText(im, str(round(cobb1, 2)),putpoint0, cv.FONT_HERSHEY_SIMPLEX, 4,
                    red, 5)
    im = cv.putText(im, str(round(cobb2, 2)), putpoint1, cv.FONT_HERSHEY_SIMPLEX, 4,
                    green, 5)
    cv.imwrite(output_file, im)
