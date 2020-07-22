import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
from math import sqrt
from math import atan
import math

img_dir = ".\img"
img_list = os.listdir(img_dir)
# img_list = list(map(lambda x: os.path.join(img_dir, x), img_list))
print(img_list)
output = ".\\test5"

yellow = (0, 255, 255)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)


def draw_upper_vertebra(vertebrae_information, color):
    for item in vertebrae_information:
        if item['location']:
            pt1 = (item['vertexes'][2][0], item['vertexes'][2][1])
            pt2 = (item['vertexes'][3][0], item['vertexes'][3][1])
            cv.line(im, pt1, pt2, color, 20)
        else:
            pt1 = (item['vertexes'][1][0], item['vertexes'][1][1])
            pt2 = (item['vertexes'][2][0], item['vertexes'][2][1])
            cv.line(im, pt1, pt2, color, 20)


def draw_turning_point(turning_location, vertebrae_information, color):
    for item in turning_location:
        location = vertebrae_information[item['previous']]
        if location['location']:
            pt1 = (location['vertexes'][2][0], location['vertexes'][2][1])
            pt2 = (location['vertexes'][3][0], location['vertexes'][3][1])
            cv.line(im, pt1, pt2, color, 20)
        else:
            pt1 = (location['vertexes'][1][0], location['vertexes'][1][1])
            pt2 = (location['vertexes'][2][0], location['vertexes'][2][1])
            cv.line(im, pt1, pt2, color, 20)


def draw_singal_vertebra(item, color, up):
    if up:
        if item['location']:
            pt1 = (item['vertexes'][2][0], item['vertexes'][2][1])
            pt2 = (item['vertexes'][3][0], item['vertexes'][3][1])
            cv.line(im, pt1, pt2, color, 20)
        else:
            pt1 = (item['vertexes'][1][0], item['vertexes'][1][1])
            pt2 = (item['vertexes'][2][0], item['vertexes'][2][1])
            cv.line(im, pt1, pt2, color, 20)
    else:
        if item['location']:
            pt1 = (item['vertexes'][0][0], item['vertexes'][0][1])
            pt2 = (item['vertexes'][1][0], item['vertexes'][1][1])
            cv.line(im, pt1, pt2, color, 20)
        else:
            pt1 = (item['vertexes'][0][0], item['vertexes'][0][1])
            pt2 = (item['vertexes'][3][0], item['vertexes'][3][1])
            cv.line(im, pt1, pt2, color, 20)


for item in tqdm(img_list):
    file = os.path.join(img_dir, item)
    output_file = os.path.join(output, item)

    im = cv.imread(file)
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

    # 椎骨的端点信息,中心点，上端椎斜率，下端椎斜率
    vertebrae_information = []
    turning_location = []

    for index in range(len(rect_list)):
        center = (int(rect_list[index][0][0]), int(rect_list[index][0][1]))
        puttex_center_point = (int(rect_list[index][0][0]) + 50, int(rect_list[index][0][1]))
        puttext_point_2 = (int(rect_list[index][0][0]) - 80, int(rect_list[index][0][1]))

        im = cv.putText(im, str(index), center, cv.FONT_HERSHEY_SIMPLEX, 2, blue, 3)
        im = cv.putText(im, str(round(rect_list[index][2], 2)), puttex_center_point, cv.FONT_HERSHEY_SIMPLEX, 3,
                        green, 3)

        # 保存椎骨顶点信息
        box = cv.boxPoints(rect_list[index])
        box = np.int0(box)
        # 判断定点01的位置，是横着的还是竖着的
        point0 = (box[0][0], box[0][1])
        point1 = (box[1][0], box[1][1])
        point2 = (box[2][0], box[2][1])
        distance_01 = sqrt(pow(point0[0] - point1[0], 2) + pow(point0[1] - point1[1], 2))
        distance_12 = sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))

        # 2 3
        # 1 0
        if distance_01 > distance_12:
            up_slope = (box[3][1] - box[2][1]) / (box[3][0] - box[2][0])
            down_slope = (box[0][1] - box[1][1]) / (box[0][0] - box[1][0])
            vertex = {'index': index, 'location': True, 'vertexes': box, 'center': center, 'up_slope': up_slope,
                      'down_slope': down_slope}
            vertebrae_information.append(vertex)
        # 1 2
        # 0 3
        else:
            up_slope = (box[2][1] - box[1][1]) / (box[2][0] - box[1][0])
            down_slope = (box[3][1] - box[0][1]) / (box[3][0] - box[0][0])
            vertex = {'index': index, 'location': False, 'vertexes': box, 'center': center, 'up_slope': up_slope,
                      'down_slope': down_slope}
            vertebrae_information.append(vertex)
        im = cv.drawContours(im, [box], 0, green, 1)

    # 找出拐点
    flag_index = [-1, -1]
    for index in range(len(rect_list)):
        if index + 1 < len(rect_list):
            if vertebrae_information[index]['location'] == False and vertebrae_information[index + 1][
                'location'] == True:
                if flag_index[1] != index:  # 去除间隔为1的状况
                    flag_index = [index, index + 1]
                    turning_index = {'previous': index, 'last': index + 1}
                    turning_location.append(turning_index)
            if vertebrae_information[index]['location'] == True and vertebrae_information[index + 1][
                'location'] == False:
                if flag_index[1] != index:
                    flag_index = [index, index + 1]
                    turning_index = {'previous': index, 'last': index + 1}
                    turning_location.append(turning_index)

    # 找出上下端锥绝对值最大的椎骨
    # 并计算最大cobb角
    slope_decline = vertebrae_information.copy()
    slope_decline.sort(key=lambda x: x['up_slope'], reverse=True)
    end_index = len(slope_decline) - 1
    max_cobb = []

    if slope_decline[0]['center'][1] > slope_decline[end_index]['center'][1]:
        draw_singal_vertebra(slope_decline[end_index], red, True)  # 上端椎
        draw_singal_vertebra(slope_decline[0], red, False)  # 下端椎

        k1 = slope_decline[end_index]['up_slope']
        k2 = slope_decline[0]['down_slope']
        angle = math.degrees(atan(abs((k2 - k1) / (1 + k1 * k2))))
        cobb = {'up_id': slope_decline[end_index]['index'], 'bottom_id': slope_decline[0]['index'], 'cobb_angle': angle}
        max_cobb.append(cobb)

        # 标注出角度
        put_point = (slope_decline[0]['center'][0] - 500, slope_decline[0]['center'][1] - 600)
        im = cv.putText(im, str(round(angle, 2)), put_point, cv.FONT_HERSHEY_SIMPLEX, 4,
                        red, 4)
    else:
        draw_singal_vertebra(slope_decline[end_index], red, False)  # 下端椎
        draw_singal_vertebra(slope_decline[0], red, True)  # 上端椎

        k1 = slope_decline[0]['up_slope']
        k2 = slope_decline[end_index]['down_slope']
        angle = math.degrees(atan(abs((k2 - k1) / (1 + k1 * k2))))
        cobb = {'up_id': slope_decline[0]['index'], 'bottom_id': slope_decline[end_index]['index'], 'cobb_angle': angle}
        max_cobb.append(cobb)

        put_point0 = (slope_decline[end_index]['center'][0] - 500, slope_decline[end_index]['center'][1])
        # im = cv.putText(im, str(slope_decline[0]['index']), put_point0, cv.FONT_HERSHEY_SIMPLEX, 4,
        #                 green, 4)
        put_point = (slope_decline[1]['center'][0] - 500, slope_decline[1]['center'][1] - 600)
        im = cv.putText(im, str(round(angle, 2)), put_point, cv.FONT_HERSHEY_SIMPLEX, 4,
                        red, 4)

    # 根据最大cobb角计算其余角度
    for item in max_cobb:
        # 有一个在边界处的情况
        up_id = int(item['up_id'])
        bt_id = int(item['bottom_id'])
        print(up_id, bt_id)
        ver_top = vertebrae_information[0]
        ver_end = vertebrae_information[len(vertebrae_information) - 1]

        if item['up_id'] == 0 or item['bottom_id'] == (len(vertebrae_information) - 1):
            if item['up_id'] == 0:
                if item['bottom_id'] == (len(max_cobb) - 1):
                    print("整个是一个弯")  # type0
                else:
                    print("剩下的下部分找两个弯")  # type1
                    # 下半部分重新排序
                    rest_series = vertebrae_information.copy()[bt_id - 1:]
                    rest_series.sort(key=lambda x: x['up_slope'], reverse=True)
                    bottom_id = len(rest_series) - 1

                    new_vertebrae = [rest_series[0], rest_series[bottom_id], ver_end]
                    new_vertebrae.sort(key=lambda x: x['center'][1])
                    # 计算下半部cobb
                    k1 = new_vertebrae[0]['up_slope']
                    k2 = new_vertebrae[1]['down_slope']
                    k3 = new_vertebrae[1]['up_slope']
                    k4 = new_vertebrae[2]['down_slope']
                    cobb1 = math.degrees(atan(abs((k2 - k1) / (1 + k1 * k2))))
                    cobb2 = math.degrees(atan(abs((k4 - k3) / (1 + k4 * k3))))

                    texPoint1 = (new_vertebrae[0]['center'][0] - 500, new_vertebrae[0]['center'][1])
                    im = cv.putText(im, str(round(cobb1, 2)), texPoint1, cv.FONT_HERSHEY_SIMPLEX, 4,
                                    yellow, 4)
                    texPoint2 = (new_vertebrae[2]['center'][0] - 500, new_vertebrae[2]['center'][1])
                    im = cv.putText(im, str(round(cobb2, 2)), texPoint2, cv.FONT_HERSHEY_SIMPLEX, 4,
                                    green, 4)

                    draw_singal_vertebra(new_vertebrae[0], yellow, True)
                    draw_singal_vertebra(new_vertebrae[1], yellow, False)
                    draw_singal_vertebra(new_vertebrae[1], green, True)
                    draw_singal_vertebra(new_vertebrae[2], green, False)


            else:
                print("上部分找两个弯")  # type2
                # 上半部分重新排序
                rest_series = vertebrae_information.copy()[:up_id + 1]
                rest_series.sort(key=lambda x: x['up_slope'], reverse=True)
                bottom_id = len(rest_series) - 1

                new_vertebrae = [rest_series[0], rest_series[bottom_id], ver_top]
                new_vertebrae.sort(key=lambda x: x['center'][1])

                # 计算上半部分cobb
                k1 = new_vertebrae[0]['up_slope']
                k2 = new_vertebrae[1]['down_slope']
                k3 = new_vertebrae[1]['up_slope']
                k4 = new_vertebrae[2]['down_slope']
                cobb1 = math.degrees(atan(abs((k2 - k1) / (1 + k1 * k2))))
                cobb2 = math.degrees(atan(abs((k4 - k3) / (1 + k4 * k3))))

                texPoint1 = (new_vertebrae[0]['center'][0] - 500, new_vertebrae[0]['center'][1])
                im = cv.putText(im, str(round(cobb1, 2)), texPoint1, cv.FONT_HERSHEY_SIMPLEX, 4,
                                yellow, 4)
                texPoint2 = (new_vertebrae[2]['center'][0] - 500, new_vertebrae[2]['center'][1])
                im = cv.putText(im, str(round(cobb2, 2)), texPoint2, cv.FONT_HERSHEY_SIMPLEX, 4,
                                green, 4)

                draw_singal_vertebra(new_vertebrae[0], yellow, True)
                draw_singal_vertebra(new_vertebrae[1], yellow, False)
                draw_singal_vertebra(new_vertebrae[1], green, True)
                draw_singal_vertebra(new_vertebrae[2], green, False)

        else:
            print("在中间, 上下两段找弯")  # type3
            # 截取两边的分别计算
            top_series = vertebrae_information.copy()[:up_id + 1]
            bottom_series = vertebrae_information.copy()[bt_id - 1:]
            tp_end = len(top_series) - 1
            bt_end = len(bottom_series) - 1

            top_series.sort(key=lambda x: x['up_slope'], reverse=True)
            bottom_series.sort(key=lambda x: x['up_slope'], reverse=True)

            new_vertebrae1 = [top_series[0], top_series[tp_end]]
            new_vertebrae1.sort(key=lambda x: x['center'][1])
            new_vertebrae2 = [bottom_series[0], bottom_series[bt_end]]
            new_vertebrae2.sort(key=lambda x: x['center'][1])

            # 计算下半部cobb
            k1 = new_vertebrae1[0]['up_slope']
            k2 = new_vertebrae1[1]['down_slope']
            k3 = new_vertebrae2[0]['up_slope']
            k4 = new_vertebrae2[1]['down_slope']
            cobb1 = math.degrees(atan(abs((k2 - k1) / (1 + k1 * k2))))
            cobb2 = math.degrees(atan(abs((k4 - k3) / (1 + k4 * k3))))

            texPoint1 = (new_vertebrae1[0]['center'][0] - 500, new_vertebrae1[0]['center'][1])
            im = cv.putText(im, str(round(cobb1, 2)), texPoint1, cv.FONT_HERSHEY_SIMPLEX, 4,
                            yellow, 4)
            texPoint2 = (new_vertebrae2[1]['center'][0] - 500, new_vertebrae2[1]['center'][1])
            im = cv.putText(im, str(round(cobb2, 2)), texPoint2, cv.FONT_HERSHEY_SIMPLEX, 4,
                            green, 4)

            draw_singal_vertebra(new_vertebrae1[0], yellow, True)
            draw_singal_vertebra(new_vertebrae1[1], yellow, False)
            draw_singal_vertebra(new_vertebrae2[0], green, True)
            draw_singal_vertebra(new_vertebrae2[1], green, False)

    cv.imwrite(output_file, im)
