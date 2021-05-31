import cv2
import numpy as np
# 创建一个200*200的黑色空白图像
img = np.zeros((200, 200), dtype=np.uint8)
# 利用numpy数组在切片上赋值的功能放置一个白色方块
img[50:150, 50:150] = 255

original_img = cv2.imread(r"C:\Users\zy199\Desktop\2344d73bf4e005e9a2be8e94479851b3.jpg")
img = cv2.cvtColor(original_img,cv2.COLOR_BGR2GRAY,cv2.THRESH_BINARY)

# 对图像进行二值化操作
# threshold(src, thresh, maxval, type, dst=None)
# src是输入数组，thresh是阈值的具体值，maxval是type取THRESH_BINARY或者THRESH_BINARY_INV时的最大值
# type有5种类型,这里取0： THRESH_BINARY ，当前点值大于阈值时，取maxval，也就是前一个参数，否则设为0
# 该函数第一个返回值是阈值的值，第二个是阈值化后的图像
ret, thresh = cv2.threshold(img, 100, 255, 0)

# findContours()有三个参数：输入图像，层次类型和轮廓逼近方法
# 该函数会修改原图像，建议使用img.copy()作为输入
# 由函数返回的层次树很重要，cv2.RETR_TREE会得到图像中轮廓的整体层次结构，以此来建立轮廓之间的‘关系’。
# 如果只想得到最外面的轮廓，可以使用cv2.RETE_EXTERNAL。这样可以消除轮廓中其他的轮廓，也就是最大的集合
# 该函数有两个返回值：修改后的图像，图像的轮廓，它们的层次
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0, 255, 0), 10)
cv2.namedWindow('contours',cv2.WINDOW_NORMAL)
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()

