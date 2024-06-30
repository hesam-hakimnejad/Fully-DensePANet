import cv2

img = cv2.imread('rendered_image(3).png')

img = cv2.resize(img, (1024,1024))

cv2.imwrite('kk.png', img)