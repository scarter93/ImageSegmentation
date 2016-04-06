/**
/ Modified from:
/ https://jayrambhia.wordpress.com/2012/09/20/roi-bounding-box-selection-of-mat-images-in-opencv/
/
/
**/

#include "generate_rect.h"

Point point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }
     
    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        /* mouse dragged. ROI being selected */
        Mat img1 = img.clone();
        point2 = Point(x, y);
        rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        imshow("image", img1);
    }
     
    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        rect = Rect(point1.x,point1.y,x-point1.x,y-point1.y);
        drag = 0;
        //roiImg = img(rect);
    }
     
    if (event == CV_EVENT_LBUTTONUP)
    {
       /* ROI selected */
        select_flag = 1;
        drag = 0;
    }
}

Rect generate_rect(Mat img_in){
	img = img_in;
	int k;

	rectangle(img, rect, CV_RGB(255, 0, 0), 3, 8, 0);
	cvSetMouseCallback("image", mouseHandler, NULL);
	imshow("image", img);
	while(1)
    {
		cvSetMouseCallback("image", mouseHandler, NULL);

        k = waitKey(10);
		
        if (select_flag == 1)
        {
            break;
        }
    }
	cvSetMouseCallback("image", NULL, NULL);
	destroyWindow("image");
	//rect = Rect(point1.x,point1.y,point2.x-point1.x,point2.y-point1.y);
	std::cout << rect << endl;
	return rect;
}


