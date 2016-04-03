
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdint.h>
#include "opencv2/ml/ml.hpp"

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

	int numClusters = 2;
	EM em_object(numClusters);

	Mat feature_mat = image.clone().reshape(1,3).t();
	feature_mat.convertTo(feature_mat,CV_64F);
	
	mexPrintf("feature_mat.rows,feature_mat.cols = %d,%d\n", feature_mat.rows,feature_mat.cols);

	Mat init_means = Mat::zeros(numClusters,3, CV_64F);

}