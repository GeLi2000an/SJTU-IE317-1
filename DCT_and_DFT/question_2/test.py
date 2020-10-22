import cv2
import numpy as np
from transforms import my_dct, my_idct, dct, idct
import argparse

img = cv2.imread(r'../figures/car.jpg')

parser = argparse.ArgumentParser()
parser.add_argument("-img_type", type=str, help="image type", default="RGB")
args = parser.parse_args()
img_type = args.img_type

img_r = my_idct(my_dct(img))
if img_type != "RGB":
	img_r = img_r[:, :, 0]
img = np.array(img, np.uint8)
img_r = np.array(img_r/255, np.float32)

cv2.imshow('1', img)
cv2.imshow('2', img_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("../figures_after/QiuSHuzhen_after_dct_keep_1_rgb.png", img_r)