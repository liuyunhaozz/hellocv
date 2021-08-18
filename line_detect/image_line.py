import cv2
import numpy as np
import math

# 霍夫直线检测
def line_image(gray):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLines(gray, 1, np.pi/180, 40)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
# 霍夫直线检测P
def line_imageP(gray):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 40,40,30)
    try:
        for x1, y1, x2, y2 in lines[0]:
            x1,y1,x2,y2 = Extend_line(x1,y1,x2,y2,160,120,0)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            num = angle(x1, y1, x2, y2)
            cv2.putText(image, str(num), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    except:
        return

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
	# erosion2 = cv2.erode(dilation, element1, iterations = 1)
	# 再次膨胀，让轮廓明显一些
	# dilation2 = cv2.dilate(erosion2, element2,iterations = 2)
	# cv2.imshow('dilation2',dilation2)
	# cv2.waitKey(0)
	return dilation

# 输出斜率
def angle(x1,y1,x2,y2):
    if(x1==x2):
        return 0

    #计算直角斜边
    a = x1-x2
    b = abs(y1-y2)
    c = math.sqrt(a**2+b**2)

    #计算斜率
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))

    if(y1>y2):
        return round(-A,2)
        
    return round(A,2)

# 直线延长
def Extend_line(x1, y1, x2, y2, x, y, flag):
    if flag == 1:
        if y1 == y2:
            return 0, y1, x, y2
        else:
            k = (y2 - y1) / (x2 - x1)
            b = (x1*y2-x2*y1)/(x1-x2)
            x3 = 0
            y3 = b
            x4 = x
            y4 = int(k * x4+b)
        return x3, y3, x4, y4
    else:
        if x1 == x2:
            return x1, 0, x2, y
        else:
            k = (y2 - y1) / (x2 - x1)
            b = (x1 * y2 - x2 * y1) / (x1 - x2)
            y3 = 0
            x3 = int(-1*b/k)
            y4 = y
            x4 = int((y4-b)/k)
            return x3, y3, x4, y4

if __name__ == '__main__':
    image = cv2.imread('./1.jpg',0)
    # cap = cv2.VideoCapture('1.mp4')

    # # 创建VideoWriter类对象
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # out = cv2.VideoWriter('outVideo.mp4', fourcc, fps, size)
    # while True:    
        # ret, image =cap.read()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = preprocess(gray)
    # img=cv2.Canny(img, 100, 200)
    pts=np.array([[160,80],[160,0],[0,0],[0,80],[80,40]])
    cv2.fillPoly(img, [pts], (0, 0, 0)) #填充
    line_imageP(img)
    # out.write(image)
    cv2.imshow("img", img)
    cv2.imshow("image", image)
    # cv2.waitKey(0)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    # if k == 27:
    #     break
    # cv2.waitKey(0)
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()