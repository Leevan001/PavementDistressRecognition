import numpy as np
import cv2

img_url = "./imgs/290_104631683.jpg"
image = cv2.imread(img_url)

#读取图片
src = image

#设置卷积核
kernel = np.array([
    [1, 2, 1],
    [1, 1, 1],
    [-1, -2, -1]
], dtype=np.int8)

result = cv2.filter2D(src, -1, kernel)
result = cv2.medianBlur(result, ksize=3)

# 加载图像
# image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# image = cv2.resize(image, np.array((100, 100)))

#在这里设置参数
# winSize = (128,128)
# blockSize = (64,64)
# blockStride = (8,8)
# cellSize = (16,16)
# nbins = 9

#定义对象hog，同时输入定义的参数，剩下的默认即可
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
# winStride = (8,8)
# padding = (8,8)
# test_hog = hog.compute(image, winStride, padding).reshape((-1,))
# print(test_hog.shape)


#等待显示
cv2.namedWindow("src", 0)
cv2.resizeWindow("src", src.shape[1] // 4, src.shape[0] // 4)

cv2.namedWindow("result", 0)
cv2.resizeWindow("result", result.shape[1] // 4, result.shape[0] // 4)


#显示图像
cv2.imshow("src", src)
cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

