import cv2

img_2 = cv2.imread("2.jpg")
img = img_2[75:, 20:, :]

cv2.imshow('2', img)
cv2.imwrite('../figures/GalGadot.jpg', img)