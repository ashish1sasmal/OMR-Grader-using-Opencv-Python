import cv2
import numpy as np
import imutils
from imutils import contours

cv2.namedWindow("OMR",cv2.WINDOW_NORMAL)

img = cv2.imread("test/omr.jpg")

# img = img[226:680,32:130]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.Canny(blurred, 75, 200)

cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()

ANS = {0:0 , 1:1 , 2:2 , 3:1 , 4:0 , 5:2 }

if len(cnts)>0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break


thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

questionCnts = []
correct = 0
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
        # cv2.drawContours(output,[c],-1,(0,255,0), 3)

questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

f=[]

for (q, i) in enumerate(np.arange(0, len(questionCnts), 3)):
    cnts = contours.sort_contours(questionCnts[i:i + 3])[0]
    # cv2.drawContours(output,cnts,j,(0,255,0), 3)
    bubbled = None

    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total < bubbled[0]:
            bubbled = (total, j)

    color = (0, 0, 255)
    k = ANS[q]

    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(img, [cnts[k]], -1, color, 3)

score = (correct / 6.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(img, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imwrite("output/result.jpg",img)
cv2.imshow("OMR", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
