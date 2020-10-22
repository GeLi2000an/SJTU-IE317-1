import cv2
import numpy as np
from transforms import my_dct, my_idct, my_dft, my_idft
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-method", type=str, help=None, default="DFT")
parser.add_argument("-theme", type=str, help="image theme", default="QiuShuzhen")
args = parser.parse_args()
method = args.method
theme = args.theme

if method == "DFT":
	forward, backward = my_dft, my_idft
else:
	forward, backward = my_dct, my_idct

if theme == "QiuShuzhen":
	img_name = '../figures/QiuShuzhen.jpg'
elif theme == 'LinDan':
	img_name = '../figures/LinDan.jpg'
else:
	img_name = '../figures/car.jpg'

img = cv2.imread(img_name)
img_r = backward(forward(img))

img = np.array(img, np.uint8)
img_r = np.array(img_r, np.uint8)

cv2.imshow('1', img)
cv2.imshow('2', img_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("../figures_after/QiuShuzhen_after_dct.jpg", img_r)