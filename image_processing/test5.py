import cv2
import imutils

img = cv2.imread(r"C:\Users\zy199\Desktop\fc25e8ed0ae709ab91486b160e1c9144.jpeg")
img = cv2.resize(img, (640, 480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 13, 15, 15)
# ret, binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(gray, 30, 120,3)            # 边缘检测

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
cnts = cnts[1] if imutils.is_cv3() else cnts[0]  # 判断是opencv2还是opencv3
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)                                       # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02*peri, True)           # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            docCnt = approx
            break
print(docCnt)
for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))

# cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)