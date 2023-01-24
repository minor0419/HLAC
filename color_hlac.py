import numpy as np
import extract_hlac
import time
import cv2
import pandas as pd


def color_hlac(cliped_frame):
    df_hlac = pd.DataFrame()

    # ゼロ埋めの画像配列
    if len(cliped_frame.shape) == 3:
        height, width, channels = cliped_frame.shape[:3]
    else:
        height, width = cliped_frame.shape[:2]
        channels = 1
    zeros = np.zeros((height, width), cliped_frame.dtype)

    # RGB分離
    img_blue_c1, img_green_c1, img_red_c1 = cv2.split(cliped_frame)

    time_sta = time.time()

    print(f'{j:4d} blue')
    img_blue_c1 = cv2.merge((img_blue_c1, zeros, zeros))
    img_blue_c1 = cv2.cvtColor(img_blue_c1, cv2.COLOR_BGR2GRAY)
    ret, img_blue_c1 = cv2.threshold(img_blue_c1, 0, 255, cv2.THRESH_OTSU)
    hlac_blue = extract_hlac(img_blue_c1)

    print(f'{j:4d} green')
    img_green_c1 = cv2.merge((img_green_c1, zeros, zeros))
    img_green_c1 = cv2.cvtColor(img_green_c1, cv2.COLOR_BGR2GRAY)
    ret, img_green_c1 = cv2.threshold(img_green_c1, 0, 255, cv2.THRESH_OTSU)
    hlac_green = extract_hlac(img_green_c1)

    print(f'{j:4d} red')
    img_red_c1 = cv2.merge((img_red_c1, zeros, zeros))
    img_red_c1 = cv2.cvtColor(img_red_c1, cv2.COLOR_BGR2GRAY)
    ret, img_red_c1 = cv2.threshold(img_red_c1, 0, 255, cv2.THRESH_OTSU)
    hlac_red = extract_hlac(img_red_c1)

    time_end = time.time()
    tim = time_end - time_sta
    print(tim)

    color_hlac = np.concatenate([hlac_blue, hlac_green, hlac_red])
    hlac_series = pd.Series(color_hlac)

    df_hlac['HLAC'] = hlac_series
