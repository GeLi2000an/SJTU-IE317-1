import cv2
import numpy as np
from question_1.transforms import my_dft, my_idft

img_1 = cv2.imread(r'../figures/LinDan.jpg')
img_1_r = my_idft(my_dft(img_1))

img_1 = np.array(img_1, np.uint8)
img_1_r = np.array(img_1_r, np.uint8)


img_2 = cv2.imread(r'../figures/QiuShuzhen.jpg')
img_2_r = my_idft(my_dft(img_2))

img_2 = np.array(img_2, np.uint8)
img_2_r = np.array(img_2_r, np.uint8)

cv2.imshow('1', img_1)
cv2.imshow('2', img_2)
cv2.imshow('1_r', img_1_r)
cv2.imshow('2_r', img_2_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
