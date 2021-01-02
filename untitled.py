import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("bars")

def hello(x):
    pass

cv2.createTrackbar("upper_hue","bars",130,180,hello)
cv2.createTrackbar("upper_sat","bars",255,255,hello)
cv2.createTrackbar("upper_val","bars",255,255,hello)
cv2.createTrackbar("lower_hue","bars",70,180,hello)
cv2.createTrackbar("lower_sat","bars",50,255,hello)
cv2.createTrackbar("lower_val","bars",50,255,hello)

while True:
    cv2.waitKey(1000)
    ret,init_frame = cap.read()
    if ret:
        break
while True:
    _,frame = cap.read()
    inspect = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    upper_hue = cv2.getTrackbarPos("upper_hue", "bars")
    upper_sat = cv2.getTrackbarPos("upper_sat", "bars")
    upper_val = cv2.getTrackbarPos("upper_val", "bars")
    lower_hue = cv2.getTrackbarPos("lower_hue", "bars")
    lower_sat = cv2.getTrackbarPos("lower_sat", "bars")
    lower_val = cv2.getTrackbarPos("lower_val", "bars")
    upper_hsv = np.array([upper_hue,upper_sat,upper_val])
    lower_hsv = np.array([lower_hue,lower_sat,lower_val])

    kernel = np.ones((3,3),np.uint8)

    mask = cv2.inRange(inspect ,lower_hsv,upper_hsv)
    mask = cv2.medianBlur(mask,3)
    mask = cv2.dilate(mask,kernel)
    mask_inv = 255-mask

    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]

    b = cv2.bitwise_and(mask_inv, b)
    g = cv2.bitwise_and(mask_inv, g)
    r = cv2.bitwise_and(mask_inv, r)
    frame_inv = cv2.merge((b,g,r))
    b = init_frame[:, :, 0]
    g = init_frame[:, :, 1]
    r = init_frame[:, :, 2]
    b = cv2.bitwise_and(mask, b)
    g = cv2.bitwise_and(mask, g)
    r = cv2.bitwise_and(mask, r)
    blanket_area = cv2.merge((b,g,r))
    final = cv2.bitwise_or(blanket_area,frame_inv)

    cv2.imshow("Invisble cloak",final)
    cv2.imshow("Original frame",frame)

    if (cv2.waitKey(1) == 27):
        break
cv2.destroyAllWindows()
cap.release()
