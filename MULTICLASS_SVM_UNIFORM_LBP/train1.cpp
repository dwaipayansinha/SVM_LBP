#include "global.h"
#include <string.h>
#include <opencv2/ml.hpp>

using namespace cv::ml;

#define ELEM(start,step,xpos,ypos) (*((uchar*)(start+step*(ypos)+(xpos)*sizeof(uchar))))

Size size1(270, 270);

Box B;
double max1, mouth_center_height, nose_center_height, eyes_center_height, nose_end_height, mouth_start_height;

Mat frame_gray, frame, frame1, faceROI, faceROI1, faceROI2, image_roi, temp1, temp, trainingData, testingData, trainingLabel;

std::vector<Rect> faces, eyes, nose, mouth;

vector<cv::String> fn;

vector<cv::Mat> data1[7];

String face_cascade_name;
String eyes_cascade_name;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int row_blocks = 9, col_blocks = 9;

int Box::train1()
{
	cout << "Training is going on! Please Wait!" << endl;
	/*std::vector<std::string> names = { "Anger", "Disgust", "Fear", "Joy", "Neutral" ,"Sad", "Surprise" };
	std::vector<std::string> path{ "../CK2/Anger/", "../CK2/Disgust/", "../CK2/Fear/", "../CK2/Joy/", "../CK2/Neutral", "../CK2/Sad", "../CK2/Surprise" };*/

	std::vector<std::string> names = { "Anger", "Disgust", "Fear", "Joy", "Neutral" };
	std::vector<std::string> path{ "../CK2/Anger/", "../CK2/Disgust/", "../CK2/Fear/", "../CK2/Joy/", "../CK2/Neutral" };

	trainingData = NULL;
	trainingLabel = NULL;
	// read each file in the folder
	for (int i = 0; i < names.size(); i++)
	{
		cv::glob(path[i], fn, true);
		int count = 0;
		for (int k = 0; k < fn.size(); k++)
		{
			cv::Mat im = cv::imread(fn[k], CV_32FC1);
			if (im.empty()) continue; //only proceed if successful
			data1[i].push_back(im);
			++count;

		}

	}

	cout << "Image path has been set!" << endl;

	//// Set SVM parameters

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	cout << "SVM has been initialized!" << endl;

	for (int i = 0; i < path.size(); i++)
	{
		B.labelling(i, trainingData, trainingLabel, data1[i], i);
	}
	/*Mat trainingData1(trainingData.rows, trainingData.cols, CV_32FC1, cv::Scalar(0));
	trainingData.copyTo(trainingData1);*/

	trainingLabel.convertTo(trainingLabel, CV_32S);

	svm->train(trainingData, ROW_SAMPLE, trainingLabel);
	svm->save("SVM_LBP.xml");
	cout << "SVM has beed trained and stored" << endl;
	return 0;
}




int Box::labelling(int i, Mat &trainingData, Mat &trainingLabel, const std::vector<cv::Mat> &data_current, const float label)
{

	int R = 1;
	int P = 8;


	face_cascade_name = "../haarcascade_frontalface_alt.xml";
	/*eyes_cascade_name = "../haarcascade_mcs_eyepair_big.xml";*/
	Mat trainingDataTemp, trainingDataTemp1;

	try
	{

		if (!face_cascade.load(face_cascade_name))
		{
			throw std::invalid_argument("Error Loading haarcascade_frontalface_alt.xml");
		}
		/*if (!eyes_cascade.load(eyes_cascade_name))
		{
		throw std::invalid_argument("Error Loading haarcascade_mcs_eyepair_big.xml");
		}*/


		for (int t = 0; t < data_current.size(); t++)
		{
			int bins = (P*(P - 1) + 3);
			frame = data_current[t];
			resize(frame, frame, size1);
			int size_of_row_blocks = frame.rows / row_blocks;
			int size_of_col_blocks = frame.cols / col_blocks;
			int init_x = 0, init_y = 0;
			Mat block;
			cv::Mat feature_hist(1, ((row_blocks*col_blocks))*(bins), CV_32FC1, Scalar(0));

			//imshow("frame", frame);
			//waitKey(0);



			/*	eyes_cascade.detectMultiScale(frame, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));*/

			int prevsize = 0;
			int size = bins - 1;
			cv::Mat eye_temp_hist(1, bins, CV_32FC1, Scalar(0));
			cv::Mat block_temp_hist(1, bins, CV_32FC1, Scalar(0));



			//for (size_t j = 0; j < eyes.size(); j++)
			//{
			//	int kk = eyes.size() - 1;
			//	if (kk == eyes.size() - 1)
			//	{
			//		//rectangle(frame, Point(eyes[j].x, eyes[j].y), Point(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height), Scalar(255, 0, 0), 1, 8, 0);
			//		Mat eye_temp;

			//		eye_temp = frame(eyes.at(j));

			//		eye_temp_hist = getLBPu2Hist(eye_temp, P, R);
			//		normalizeHist(eye_temp_hist);
			//	    eye_temp_hist.colRange(0, size).copyTo(feature_hist.colRange(prevsize, prevsize + size));
			//		
			//	}
			//  }


			for (int i = 0; i < row_blocks; i++)
			{

				for (int j = 0; j < col_blocks; j++)
				{
					Rect roi = Rect(init_x, init_y, size_of_row_blocks - 1, size_of_col_blocks - 1);
					block = frame(roi);
					/*rectangle(frame, Point(init_x, init_y), Point(init_x + size_of_row_blocks, init_y + size_of_col_blocks), Scalar(255, 0, 0), 1, 8, 0);
					imshow("frame", frame);
					waitKey(0);*/
					block_temp_hist = getLBPu2Hist(block, P, R);
					normalizeHist(block_temp_hist);

					block_temp_hist.colRange(0, size).copyTo(feature_hist.colRange(prevsize, prevsize + size));
					prevsize = prevsize + size + 1;
					init_x = init_x + size_of_col_blocks;

				}
				init_x = 0;
				init_y = init_y + size_of_row_blocks;
			}

			trainingLabel.push_back(label);
			trainingData.push_back(feature_hist);
		}
	}


	catch (Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		getchar();
	}
	catch (runtime_error& ex) {
		cerr << "Runtime Error:\n" << ex.what();
		getchar();
	}
	catch (const std::invalid_argument& e)
	{
		cout << e.what();
		getchar();
	}
	return 0;
}



Mat Box::getLBPu2Hist(cv::Mat &frame, int P, int R)
{
	unsigned bins = (unsigned)(P*(P - 1) + 3);
	cv::Mat LBPu2Hist(1, bins, CV_32FC1, Scalar(0));
	unsigned uniform[] = { 0 ,1 ,2 , 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112,120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255 };
	cv::Point *nbrTable;
	nbrTable = new Point[8];
	nbrTable[0] = Point(1, 0);
	nbrTable[1] = Point(1, -1);
	nbrTable[2] = Point(0, -1);
	nbrTable[3] = Point(-1, -1);
	nbrTable[4] = Point(-1, 0);
	nbrTable[5] = Point(-1, 1);
	nbrTable[6] = Point(0, 1);
	nbrTable[7] = Point(1, 1);


	for (int yc = R; yc < frame.rows - R; yc++)
	{
		for (int xc = R; xc < frame.cols - R; xc++)
		{

			unsigned char thresh = ELEM(frame.data, frame.step, xc, yc);
			//cout <<"position 0   "<< (int)frame.at<uchar>(0,0) << "\t"<<"frame data  " << (int)frame.at<uchar>(xc,yc) << endl;
			unsigned V = 0;
			for (int i = P - 1; i >= 0; i--)
			{

				int xp = xc + int(R * nbrTable[i].x);
				int yp = yc + int(R * nbrTable[i].y);
				/*cout << (int)thresh <<"\t"<< (int)frame.at<uchar>(yp, xp) << endl;*/
				bool vp = (ELEM(frame.data, frame.step, xp, yp) >= thresh ? 1 : 0);
				V = (V << 1) | int(vp);
			}
			/*cout << "\n" << V;
			int y;
			cin >> y;*/

			//UNIFORM LBP
			bool exists = std::find(std::begin(uniform), std::end(uniform), V) != std::end(uniform);
			int x = std::distance(uniform, std::find(uniform, uniform + 58, V));
			//cout<<"x  "<<x<<endl;
			//cout << V << "\t" << exists << endl;
			if (!exists)
				(*((float*)(LBPu2Hist.data + (bins) * sizeof(float))))++;
			else
				(*((float*)(LBPu2Hist.data + (x) * sizeof(float))))++;

		}
		/*int y;
		cin >> y;*/

	}
	return LBPu2Hist;
}

void Box::normalizeHist(cv::Mat &matG)
{
	float sum = 0;
	for (int i = 0; i < matG.cols; i++)
	{
		//cout << (*((float*)(matG.data + (i) * sizeof(float)))) << endl;
		//cout << *((float*)(matG.data))<<endl;
		sum += (*((float*)(matG.data + (i) * sizeof(float))));
	}
	if (!sum) return;
	for (int i = 0; i < matG.cols; i++)
	{

		(*((float*)(matG.data + (i) * sizeof(float)))) = (*((float*)(matG.data + (i) * sizeof(float)))) / sum;
		//cout << (*((float*)(matG.data + (i) * sizeof(float)))) << endl;
	}

}
