#include "mex.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "matrix.h"
#include <stdint.h>

using namespace cv;
using namespace std;

void mexFunction(int nlhs,  mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    if (nrhs != 2){
        mexErrMsgTxt("Wrong number of input arguments.");
		return;
    }
    if (nlhs != 1){
        mexErrMsgTxt("Wrong number of output arguments.");
		return;
    }
    if (!mxIsChar(prhs[0])){
         mexErrMsgTxt("First input is not a char.");
		 return;
     }
    if(!mxIsDouble(prhs[1])){
        mexErrMsgTxt("Second input is not a double.");
		return;
    }
    if(mxGetM(prhs[1]) != 1){
        mexErrMsgTxt("M != 1.");
		return;
    }
    if(mxGetN(prhs[1]) != 4){
        mexErrMsgTxt("N != 4.");
		return;
    }
    int M_rect = mxGetM(prhs[1]);
    int N_rect = mxGetN(prhs[1]);

    std::string img(mxArrayToString(prhs[0]));
    mexPrintf("img_name = %s\n", img);
    cv::Mat image = cv::imread(img);
	//imshow("test", image);
	//waitKey(1);

	double *rect_ptr = mxGetPr(prhs[1]);

    Rect rectangle(Point(rect_ptr[0],rect_ptr[1]),Point(rect_ptr[2],rect_ptr[3]));
    mexPrintf("rectangle.rows,rectangle.cols = %d,%d\n", rectangle.height,rectangle.width);
    //Mat feature_mat = Mat::zeros(3, image.size().width*image.size().height,CV_64F);

	Mat feature_mat = image.clone().reshape(1,3).t();
	feature_mat.convertTo(feature_mat,CV_32F);
	
	mexPrintf("feature_mat.rows,feature_mat.cols = %d,%d\n", feature_mat.rows,feature_mat.cols);
	
	Mat label = Mat::zeros(image.size().width*image.size().height,1, CV_32S);

	for(int i = 0; i < image.rows; i++){
		for( int j = 0; j < image.cols; j++){
			if(rectangle.contains(Point(i,j))){
				label.at<int>(image.cols*i + j,0) = 1;	
			}else{
				label.at<int>(image.cols*i + j,0) = 0;
			}
		}
	}
	Mat centers;
    kmeans(feature_mat, 2, label,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 5, KMEANS_PP_CENTERS, centers);

	Mat foreground_mask = Mat::zeros(image.size().height,image.size().width, CV_8UC1);

	for(int i = 0; i < image.rows; i++){
		for( int j = 0; j < image.cols; j++){
			if(label.at<int>(image.cols*i + j,0) == 1){
					foreground_mask.at<unsigned char>(i,j) = 1;
			}
		}
	}

	const size_t dims = image.size().width;

	mxArray* result = mxCreateNumericArray(image.size().height, &dims, mxUINT8_CLASS,mxREAL);



    plhs[0] = mxCreateDoubleScalar(420);
    
    return;
}