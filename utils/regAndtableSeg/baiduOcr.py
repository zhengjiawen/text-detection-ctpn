from aip import AipOcr
from PIL import Image
import cv2 as cv
import base64

import os
import numpy as np
import urllib


APP_ID = '15582892'
API_KEY = 'Xt6Sq24STPHp4zduLPHLSoKz'
SECRET_KEY = 'eYatFz0cUjc6BL8zGEEUe8EQI81Q5hQi'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def covertToWebImgType(img, imgType = '.jpg'):
    img_encode = cv.imencode(imgType, img)[1]
    data_encode = np.array(img_encode)
    frame_encode = data_encode.tostring()
    return frame_encode

def regWordByBaiduOcr(img, acc = False):
    img_encode = covertToWebImgType(img)
    if acc:
        res = client.basicAccurate(img_encode)
    else:
        res = client.basicGeneral(img_encode)
    word_result = res.get('words_result')
    if word_result != None:
        line = ''
        for item in word_result:
            line += item['words']
        return line
