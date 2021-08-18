#include<opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("D:/DUT/my_pytorch/pytorch/1.jpg");
	imshow("picture", src);
	waitKey(0);
	destroyAllWindows();
	cout << "Hello World!";
	return 0;
}