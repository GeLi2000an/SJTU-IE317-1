import cv2
import numpy as np
from question_3.transforms import my_dft, my_idft

img_1 = cv2.imread(r'../figures/road_1.png')
img_2 = cv2.imread(r'../figures/road_2.png')
dft_img_1 = my_dft(img_1)
dft_img_2 = my_dft(img_2)

dft_img_1_magnitude = np.abs(dft_img_1)
dft_img_2_magnitude = np.abs(dft_img_2)
dft_img_1_phase = np.angle(dft_img_1)
dft_img_2_phase = np.angle(dft_img_2)

dft_img_1 = dft_img_1_magnitude * np.exp(1j*dft_img_2_phase)
dft_img_2 = dft_img_2_magnitude * np.exp(1j*dft_img_1_phase)

img_1_r = my_idft(dft_img_1)
img_2_r = my_idft(dft_img_2)

img_1 = np.array(img_1, np.uint8)
img_1_r = np.array(img_1_r, np.uint8)
img_2 = np.array(img_2, np.uint8)
img_2_r = np.array(img_2_r, np.uint8)

print(img_1_r.shape)
cv2.imshow('1_original', img_1)
cv2.imshow('magnitude_1_and_phase_2', img_1_r)
cv2.imshow('2_original', img_2)
cv2.imshow('magnitude_2_and_phase_1', img_2_r)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("../figures_after/magnitude_1_and_phase_2_road_theme.png", img_1_r)
cv2.imwrite("../figures_after/magnitude_2_and_phase_1_road_theme.png", img_2_r)