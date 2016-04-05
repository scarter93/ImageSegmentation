#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdint.h>
#include <opencv2\ml\ml.hpp>
#include "generate_rect.h"

using namespace cv;
//using namespace std;

Mat my_kmeans(Mat &image, Rect &rectangle);
Mat GraphCut(Mat &image, Rect &rectangle);
Mat GMM(Mat &image, Rect &rectangle);

int main(){
	Mat image = imread("100_0109.png");
	Rect rectangle = generate_rect(image);

	if (rectangle.contains(Point(50, 50))) {
		cout << "rectangle contains this point" << endl;
	}

	//imwrite("kmeans_out.png",my_kmeans(image, rectangle));
	imwrite("graphCut_out.png", GraphCut(image, rectangle));
	return 0;
}

Mat my_kmeans(Mat &image, Rect &rectangle){
  
    //std::string img(img_name);
    //cout << "img_name = " << img_name << endl;
    //cv::Mat image = cv::imread(img_name);
	//imshow("test", image);
	//waitKey(1);

    Mat feature_mat = Mat::zeros(image.size().width*image.size().height,3,CV_32F);

	//Mat feature_mat = image.clone().reshape(1,3).t();
	//feature_mat.convertTo(feature_mat,CV_32F);
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if ( rectangle.contains(Point(j,i))) {
				//cout << "inside rect" << endl;
				feature_mat.at<float>(i*image.cols + j, 0) = image.at<cv::Vec3b>(i, j)[0];
				feature_mat.at<float>(i*image.cols + j, 1) = image.at<cv::Vec3b>(i, j)[1];
				feature_mat.at<float>(i*image.cols + j, 2) = image.at<cv::Vec3b>(i, j)[2];
			}
			else {
				feature_mat.at<float>(i*image.cols + j, 0) = 0;
				feature_mat.at<float>(i*image.cols + j, 1) = 0;
				feature_mat.at<float>(i*image.cols + j, 2) = 0;

			}
		}
	}

	//for (int i = 0; i < gray_image.rows; i++) {
	//	for (int j = 0; j < gray_image.cols; j++) {
	//		if (rectangle.contains(Point(j, i))) {
	//			//cout << "inside rect" << endl;
	//			feature_mat.at<float>(i*gray_image.cols + j, 0) = gray_image.at<uchar>(i,j);
	//			feature_mat.at<float>(i*gray_image.cols + j, 1) = gray_image.at<uchar>(i, j);
	//			feature_mat.at<float>(i*gray_image.cols + j, 2) = gray_image.at<uchar>(i, j);
	//		}
	//		else {
	//			feature_mat.at<float>(i*gray_image.cols + j, 0) = 0;
	//			feature_mat.at<float>(i*gray_image.cols + j, 1) = 0;
	//			feature_mat.at<float>(i*gray_image.cols + j, 2) = 0;
	//
	//		}
	//	}
	//}

	//imwrite("pre kmeans.png", feature_mat);
	//cout << feature_mat << endl;
	
	printf("feature_mat.rows,feature_mat.cols = %d,%d\n", feature_mat.rows,feature_mat.cols);
	
	Mat label = Mat::zeros(image.size().width*image.size().height,1, CV_32S);

	for(int i = 0; i < image.rows; i++){
		for( int j = 0; j < image.cols; j++){
			if(rectangle.contains(Point(j,i))){
				//cout << "point is located in rectangle" << endl;
				label.at<int>(image.cols*i + j,0) = 1;	
			}else{
				label.at<int>(image.cols*i + j,0) = 0;
			}
		}
	}

	printf("finished copying labels\n");

	Mat centers;
    kmeans(feature_mat, 2, label,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 5, KMEANS_PP_CENTERS, centers);

	cout << "generating feature matrix from kmeans function" << endl;
	//feature_mat.reshape(image.size().width, image.size().height);
	//imwrite("post kmeans.png", feature_mat);

	//cout << feature_mat << endl;
	//waitKey(1);

	Mat foreground_mask = Mat::zeros(image.size().height,image.size().width, CV_8UC1);
	cout << "generated and copying to foreground mask" << endl;
	for(int i = 0; i < image.rows; i++){
		for( int j = 0; j < image.cols; j++){
			if(label.at<int>(i*image.cols + j,0) == 1){
					foreground_mask.at<unsigned char>(i,j) = 255;
			}
		}
	}
	cout << "foreground mask copying complete" << endl;
	imwrite("foreground_mask.png", foreground_mask);

	Mat output;

	image.copyTo(output, foreground_mask);

	return output;
    
}

Mat GraphCut(Mat &image, Rect &rectangle) {
	Mat mask, bgdModel, fgdModel;
	grabCut(image, mask, rectangle, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);

	cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
	//Mat foreground(image.size(), CV_8UC1, cv::Scalar(255));
	imwrite("mask_GCM.png", mask);
	Mat result;
	image.copyTo(result, mask);

	return result;

}

Mat GMM(Mat &image, Rect &rectangle) {

	Mat feature_mat = Mat::zeros(image.size().width*image.size().height, 3, CV_64F);

	int numClusters = 2;
	EM em_object(numClusters);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (rectangle.contains(Point(j, i))) {
				//cout << "inside rect" << endl;
				feature_mat.at<float>(i*image.cols + j, 0) = image.at<cv::Vec3b>(i, j)[0];
				feature_mat.at<float>(i*image.cols + j, 1) = image.at<cv::Vec3b>(i, j)[1];
				feature_mat.at<float>(i*image.cols + j, 2) = image.at<cv::Vec3b>(i, j)[2];
			}
			else {
				feature_mat.at<float>(i*image.cols + j, 0) = 0;
				feature_mat.at<float>(i*image.cols + j, 1) = 0;
				feature_mat.at<float>(i*image.cols + j, 2) = 0;

			}
		}
	}

	Mat init_means = Mat::zeros(numClusters, 3, CV_64F);

	// TODO: create initial means based on foreground and background
	
	vector<Mat> cov_mats(numClusters);
	
	






}
