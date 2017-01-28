#ifndef GLOBAL_H
#define GLOBAL_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <string.h>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <opencv2/ml.hpp>



using namespace std;
using namespace cv;
using namespace cv::ml;

//extern int R = 1;
////LBP Points
//extern int P = 8;

extern double max1, mouth_center_height, nose_center_height, eyes_center_height, nose_end_height, mouth_start_height;

extern Mat frame_gray, frame, frame1, faceROI, faceROI1, faceROI2, image_roi, temp1, temp, trainingData, testingData, trainingLabel;
extern std::vector<Rect> faces, eyes, nose, mouth;
//extern vector<float> trainingLabel;
//cv::Point *nbrTable;

extern vector<cv::String> fn, fn_test;
extern vector<cv::Mat> data1[7], data_test;

extern String face_cascade_name;
extern String eyes_cascade_name;
extern String nose_cascade_name;
extern String mouth_cascade_name;

extern CascadeClassifier face_cascade;
extern CascadeClassifier eyes_cascade;
extern CascadeClassifier nose_cascade;
extern CascadeClassifier mouth_cascade;

class Box
{
public:
	//Box();
	//~Box();
	int train1();
	int test();
	//int labelling(int i, Mat &trainingData, vector<float> &trainingLabel, const vector<String> &names, const std::vector<cv::Mat> &data_current, const float label);
	int labelling(int i, Mat &trainingData, Mat &trainingLabel, const std::vector<cv::Mat> &data_current, const float label);
	int prediction(const cv::Mat &testingData);
	//cv::Mat calcHistogram(Mat &image);
	cv::Mat getLBPu2Hist(cv::Mat &frame, int P, int R);
	void normalizeHist(cv::Mat &matG);
	void calculate();
	//void init();
};

#endif // GLOBAL_H