import tesserocr
import cv2 as cv
from PIL import Image
import os


def regWordByTesserocr(img):
    image = Image.fromarray(img)
    result = tesserocr.image_to_text(image, lang='chi_sim+equ+eng')
    return result