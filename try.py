import cv2

img = cv2.imread('./test_images1/000017.png')
print(img.shape)
h, w = img.shape[:2]
cv2.rectangle(img, (int(0.11914699*w), int(0.0375188*h)), (int(0.4153133*w), int(0.7737459*h)), (0, 255, 0), 3)
cv2.rectangle(img, (int(0.4428676*w), int(0.09341463*h)), (int(0.73502827*w), int(0.8565788 *h)), (0, 255, 255), 3)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
