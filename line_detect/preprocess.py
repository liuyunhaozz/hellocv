import cv2

def preprocess(gray):
	# # 直方图均衡化
	equ = cv2.equalizeHist(gray)
	# 高斯平滑
	gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
	# 中值滤波
	median = cv2.medianBlur(gaussian, 5)
	# Sobel算子，X方向求梯度
	sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize = 3)
	# 二值化
	ret, binary = cv2.threshold(sobel, 30,255, cv2.THRESH_BINARY)
	# 膨胀和腐蚀操作的核函数
	element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
	element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
	# 腐蚀一次，去掉细节
	erosion = cv2.erode(binary, element1, iterations = 1)
	# 膨胀一次，让轮廓突出
	dilation = cv2.dilate(erosion, element2, iterations = 1)
	# 腐蚀一次，去掉细节
	erosion2 = cv2.erode(dilation, element1, iterations = 1)
	# 再次膨胀，让轮廓明显一些
	dilation2 = cv2.dilate(erosion2, element2,iterations = 2)
	cv2.imshow('dilation2', binary)
	cv2.waitKey(0)

if __name__ == '__main__':
    image = cv2.imread('1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = preprocess(gray)