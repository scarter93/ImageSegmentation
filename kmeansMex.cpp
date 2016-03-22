#include "mex.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "matrix.h"

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
    Rect rectangle(Point(M_rect-1,M_rect-1),Point(M_rect-1,N_rect-1));
    
    Mat feature_mat = Mat::zeros(3, image.size().width*image.size().height,CV_64F);

	Mat test = image.clone().reshape(1,3).t();
	//image.reshape(0,1);
	//test = test.t();

	mexPrintf("test.rows,test.cols = %d,%d\n", test.rows,test.cols);
	mexPrintf("feature_mat.rows,feature_mat.cols = %d,%d\n", feature_mat.rows,feature_mat.cols);
	//for(int i = 0; i < image.rows; i++);

    
    plhs[0] = mxCreateDoubleScalar(420);
    
    return;
}