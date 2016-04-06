#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdint.h>
#include <opencv2\ml\ml.hpp>
#include "generate_rect.h"

///Defines for options

#define TREE 0
#define TOAD 1
#define PENG 2

///Select the image to work on

//#define IMG_SELECT TREE
#define IMG_SELECT TOAD
//#define IMG_SELECT PENG


using namespace cv;


Mat my_kmeans(Mat &image, Rect &rectangle, int gray);
Mat GraphCut(Mat &image, Rect &rectangle, int lab);
Mat GMM(Mat &image, Rect &rectangle, int gray);

int main(){
	Mat image = imread("100_0109.png");
	Mat image1 = imread("b4nature_animals_land009.png");
	Mat image2 = imread("cheeky_penguin.png");
	//run tree if desired
	if (IMG_SELECT == TREE) {
		Rect rectangle = generate_rect(image);

		imwrite("tree_kmeans.png", my_kmeans(image, rectangle, 0));
		imwrite("tree_graphCut.png", GraphCut(image, rectangle, 0));
		imwrite("tree_gmm.png", GMM(image, rectangle, 0));

		imwrite("tree_kmeans_gray.png", my_kmeans(image, rectangle, 1));
		imwrite("tree_graphCut_lab.png", GraphCut(image, rectangle, 1));
		imwrite("tree_gmm_gray.png", GMM(image, rectangle, 1));
	}
	//run toad if desired
	else if (IMG_SELECT == TOAD) {
		Rect rectangle1 = generate_rect(image1);

		imwrite("toad_kmeans.png", my_kmeans(image1, rectangle1, 0));
		imwrite("toad_graphCut.png", GraphCut(image1, rectangle1, 0));
		imwrite("toad_gmm.png", GMM(image1, rectangle1, 0));

		imwrite("toad_kmeans_gray.png", my_kmeans(image1, rectangle1, 1));
		imwrite("toad_graphCut_lab.png", GraphCut(image1, rectangle1, 1));
		imwrite("toad_gmm_gray.png", GMM(image1, rectangle1, 1));

	}
	//run penguin if desired
	else if (IMG_SELECT == PENG) {
		Rect rectangle2 = generate_rect(image2);

		imwrite("penguin_kmeans.png", my_kmeans(image2, rectangle2, 0));
		imwrite("penguin_graphCut.png", GraphCut(image2, rectangle2, 0));
		imwrite("penguin_gmm.png", GMM(image2, rectangle2, 0));

		imwrite("penguin_kmeans_gray.png", my_kmeans(image2, rectangle2, 1));
		imwrite("penguin_graphCut_lab.png", GraphCut(image2, rectangle2, 1));
		imwrite("penguin_gmm_gray.png", GMM(image2, rectangle2, 1));
	}
	return 0;
}

Mat my_kmeans(Mat &image, Rect &rectangle,int gray){
 	//feature matrix
    Mat feature_mat = Mat::zeros(image.size().width*image.size().height,3,CV_32F);

	//check to see if image should be grayscale
	if (gray) {
		Mat gray;
		cvtColor(image,gray, CV_BGR2GRAY);
		cvtColor(gray, image, CV_GRAY2BGR);
	}
	//fill feature mat
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
	printf("feature_mat.rows,feature_mat.cols = %d,%d\n", feature_mat.rows,feature_mat.cols);
	
	Mat label = Mat::zeros(image.size().width*image.size().height,1, CV_32S);
	//fill label matrix
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
	//run k-means
    kmeans(feature_mat, 2, label,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 5, KMEANS_PP_CENTERS, centers);

	cout << "generating feature matrix from kmeans function" << endl;

	Mat foreground_mask = Mat::zeros(image.size().height,image.size().width, CV_8UC1);
	//generate foreground mask
	cout << "generated and copying to foreground mask" << endl;
	for(int i = 0; i < image.rows; i++){
		for( int j = 0; j < image.cols; j++){
			if(label.at<int>(i*image.cols + j,0) == 1){
					foreground_mask.at<unsigned char>(i,j) = 255;
			}
		}
	}
	cout << "foreground mask copying complete" << endl;
	//use mask and return
	Mat output;
	image.copyTo(output, foreground_mask);
	return output;
}

Mat GraphCut(Mat &image, Rect &rectangle, int lab) {
	//check to see if Lab is wanted
	if (lab) {
		Mat lab_im;
		cvtColor(image, lab_im, CV_BGR2Lab);
		lab_im.copyTo(image);
	}

	Mat mask, bgdModel, fgdModel;
	//perform graphcut
	grabCut(image, mask, rectangle, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);
	cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
	//use mask and return result
	Mat result;
	image.copyTo(result, mask);
	return result;
}

Mat GMM(Mat &image, Rect &rectangle, int gray) {
	Mat feature_mat = Mat::zeros(image.size().width*image.size().height, 3, CV_32F);
	int numClusters = 2;
	//use grayscale if desired
	if (gray) {
		Mat gray;
		cvtColor(image, gray, CV_BGR2GRAY);
		cvtColor(gray, image, CV_GRAY2BGR);
	}
	//genertae feature matrix
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (rectangle.contains(Point(j, i))) {
				//cout << "inside rect" << endl;
				feature_mat.at<float>(i*image.cols + j, 0) = image.at<cv::Vec3b>(i, j)[0];
				feature_mat.at<float>(i*image.cols + j, 1) = image.at<cv::Vec3b>(i, j)[1];
				feature_mat.at<float>(i*image.cols + j, 2) = image.at<cv::Vec3b>(i, j)[2];
			}
			else {
				feature_mat.at<float>(i*image.cols + j, 0) = image.at<cv::Vec3b>(i, j)[0];
				feature_mat.at<float>(i*image.cols + j, 1) = image.at<cv::Vec3b>(i, j)[1];
				feature_mat.at<float>(i*image.cols + j, 2) = image.at<cv::Vec3b>(i, j)[2];

			}
		}
	}
	cv::EM em_obj(2);
	Mat means = Mat::zeros(numClusters, 3, CV_64F);

	int cnt_fgd = 0;
	int cnt_bgd = 0;
	//generate means
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (rectangle.contains(Point(j, i))) {
				//cout << "inside rect" << endl;
				cnt_fgd++;
				means.at<double>(0, 0) += image.at<cv::Vec3b>(i, j)[0];
				means.at<double>(0, 1) += image.at<cv::Vec3b>(i, j)[1];
				means.at<double>(0, 2) += image.at<cv::Vec3b>(i, j)[2];
			}else {
				cnt_bgd++;
				means.at<double>(1, 0) += image.at<cv::Vec3b>(i, j)[0];
				means.at<double>(1, 1) += image.at<cv::Vec3b>(i, j)[1];
				means.at<double>(1, 2) += image.at<cv::Vec3b>(i, j)[2];
			}
		}
	}
	cout << means << endl;

	means.at<double>(0, 0) = means.at<double>(0, 0) / cnt_fgd;
	means.at<double>(0, 1) = means.at<double>(0, 1) / cnt_fgd;
	means.at<double>(0, 2) = means.at<double>(0, 2) / cnt_fgd;

	means.at<double>(1, 0) = means.at<double>(1, 0) / cnt_bgd;
	means.at<double>(1, 1) = means.at<double>(1, 1) / cnt_bgd;
	means.at<double>(1, 2) = means.at<double>(1, 2) / cnt_bgd;

	cout << means << endl;
	Mat cov;

	Mat weights = Mat::zeros(1, numClusters, CV_64F);
	//generate weights
	weights.at<double>(0, 0) = (double)cnt_fgd / (cnt_bgd + cnt_fgd);
	weights.at<double>(0, 1) = (double)cnt_fgd / (cnt_bgd + cnt_fgd);

	Mat labels, probs, likelihoods;
	//train the EM
	em_obj.trainE(feature_mat, means, noArray(), weights, likelihoods, labels, probs);

	//generate the foreground mask
	Mat foreground_mask = Mat::zeros(image.size().height, image.size().width, CV_8UC1);
	cout << "generated and copying to foreground mask" << endl;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (labels.at<int>(i*image.cols + j, 0) == 1) {
				foreground_mask.at<unsigned char>(i, j) = 255;
			}
		}
	}
	cout << "foreground mask copying complete" << endl;
	//use mask and display result
	Mat result;
	image.copyTo(result, foreground_mask);
	return result;
}
