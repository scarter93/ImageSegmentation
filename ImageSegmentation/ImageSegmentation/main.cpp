#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdint.h>
#include "generate_rect.h"

//using namespace cv;
//using namespace std;


int main(){
	Mat image = imread("100_0109.png");
	Rect rectangle = generate_rect(image);
	
	return 0;
}

//void kmeans(string img_name, Rect *rectangle){
//  
//    //std::string img(img_name);
//    cout << "img_name = " << img_name << endl;
//    cv::Mat image = cv::imread(img_name);
//	//imshow("test", image);
//	//waitKey(1);
//
//    //Mat feature_mat = Mat::zeros(3, image.size().width*image.size().height,CV_64F);
//
//	Mat feature_mat = image.clone().reshape(1,3).t();
//	feature_mat.convertTo(feature_mat,CV_32F);
//	
//	mexPrintf("feature_mat.rows,feature_mat.cols = %d,%d\n", feature_mat.rows,feature_mat.cols);
//	
//	Mat label = Mat::zeros(image.size().width*image.size().height,1, CV_32S);
//
//	for(int i = 0; i < image.rows; i++){
//		for( int j = 0; j < image.cols; j++){
//			if(rectangle.contains(Point(i,j))){
//				label.at<int>(image.cols*i + j,0) = 1;	
//			}else{
//				label.at<int>(image.cols*i + j,0) = 0;
//			}
//		}
//	}
//
//	mexPrintf("finished copying labels\n");
//
//	Mat centers;
//    kmeans(feature_mat, 2, label,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 5, KMEANS_PP_CENTERS, centers);
//	mexPrintf("generating feature matrix from kmeans function\n");
//	imwrite("post kmeans.jpg", label);
//	///waitKey(1);
//
//	Mat foreground_mask = Mat::zeros(image.size().height,image.size().width, CV_8UC1);
//	mexPrintf("generated and copying to foreground mask\n");
//	for(int i = 0; i < image.rows; i++){
//		for( int j = 0; j < image.cols; j++){
//			if(label.at<int>(image.cols*i + j,0) == 1){
//					foreground_mask.at<unsigned char>(i,j) = 1;
//			}
//		}
//	}
//	mexPrintf("foreground mask copying complete\n");
//
//	int dims[2]; 
//	dims[2] = image.size().height;
//	dims[1] = image.size().width;
//
//	mexPrintf("generating numeric array\n");
//	plhs[0]= mxCreateNumericArray(2, dims, (mxClassID)9,mxREAL);
//	mexPrintf("generated output array... filling array now\n");
//	unsigned char* data_ptr = (unsigned char*)mxGetData(plhs[0]);
//	mexPrintf("data_ptr address = %x\n", data_ptr);
//	mexPrintf("accessing mxArray pointer now\n");
//	for(int j = 0; j < image.cols; j++){
//		for( int i = 0; i < image.rows; i++){
//			//mexPrintf("accessing foreground data now\n");
//			
//			//mexPrintf("data (foreground_mask) at (%d,%d) = %d\n", i, j, foreground_mask.at<unsigned char>(i,j));
//			//data_ptr[0] = foreground_mask.at<unsigned char>(i,j);
//			
//			//data_ptr++;
//		}
//		//mexPrintf("column %d being filled\n", j);
//	}
//	mexPrintf("output array filled\n");
//
//    //plhs[0] = result;
//    
//    return;
//}