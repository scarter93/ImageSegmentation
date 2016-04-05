#ifndef _GEN__RECT_
#define _GEN__RECT_

#include<opencv2/highgui/highgui.hpp>
#include<string>
#include<iostream>

using namespace cv;
using namespace std;

Rect generate_rect(cv::Mat img_in);
void mouseHandler(int event, int x, int y, int flags, void* param);


#endif //_GEN__RECT_