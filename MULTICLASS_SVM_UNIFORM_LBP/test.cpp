#include "global.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <opencv2/ml.hpp>

using namespace cv::ml;
int Box::test()
{
	Box b;
	max1 = 0.0;
	int R = 1;
	int P = 8;
	Mat frame, frame_stream, frame_stream_gray;
	int tot_score = 0, counter = 0;

	std::string output_expression;

	std::vector<std::string> names = { "Anger", "Disgust", "Fear", "Joy", "Sad", "Surprise" };
	std::vector<std::string> path{ "D:/CK2/Anger/", "D:/CK2/Disgust/", "D:/CK2/Fear/", "D:/CK2/Joy/", "D:/CK2/Sad", "D:/CK2/Surprise" };

	face_cascade_name = "../haarcascade_frontalface_alt.xml";
	eyes_cascade_name = "../haarcascade_mcs_eyepair_big.xml";
	nose_cascade_name = "../nose.xml";
	mouth_cascade_name = "../mouth.xml";

	trainingData = NULL;
	trainingLabel = NULL;

	Size size1(270, 270);
	int row_blocks = 9, col_blocks = 9;


	try
	{
		if (!face_cascade.load(face_cascade_name))
		{
			//cout << "No frontal face model" << endl;
			throw std::invalid_argument("Error Loading haarcascade_frontalface_alt.xml");
		}
		if (!eyes_cascade.load(eyes_cascade_name))
		{
			//cout << "No eyes model" << endl;
			throw std::invalid_argument("haarcascade_mcs_eyepair_big.xml");
		}
		if (!nose_cascade.load(nose_cascade_name))
		{
			//cout << "No nose model" << endl;
			throw std::invalid_argument("Error Loading nose.xml");
		}
		if (!mouth_cascade.load(mouth_cascade_name))
		{
			//cout << "No mouth model" << endl;
			throw  std::invalid_argument("Error Loading mouth.xml");
		}

		//Extract images from database
		/*for (int i = 0; i < 6; i++)
		{
			cv::glob(path[i], fn, true);
			for (int k = 0; k < fn.size(); k++)
			{
				cv::Mat im = cv::imread(fn[k], CV_32FC1);
				if (im.empty()) continue;
				data1[i].push_back(im);
			}
		}*/


		VideoCapture vcap(0);
		if (!vcap.isOpened())
		{
			throw std::invalid_argument("Error loading camera");
		}
		int score;

		ofstream ptr;
		ptr.open("score_live.txt", 'w');

		/*for (int g = 0; g < 6; g++)
		{
		for (int t = 0; t < data1[g].size(); t++)*/
		while (1)
		{
			//frame = data1[g][t];

			//LIVE STREAM
			vcap >> frame_stream;
			if (frame_stream.empty()) //check whether the image is loaded or not
			{
				cout << "Error : Image cannot be loaded..!!" << endl;
				return -1;
			}
			//


			//LIVE STREAM
			cv::cvtColor(frame_stream, frame_stream_gray, CV_BGR2GRAY);
			face_cascade.detectMultiScale(frame_stream_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			if (faces.size())
			{
				Rect roi1 = faces.at(faces.size() - 1);
				int x = roi1.x;
				int y = roi1.y;
				int h = y + roi1.height;
				int w = x + roi1.width;
				rectangle(frame_stream, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);
				frame = frame_stream_gray(roi1);
				//
				resize(frame, frame, size1);
				int size_of_row_blocks = frame.rows / row_blocks;
				int size_of_col_blocks = frame.cols / col_blocks;
				int init_x = 0, init_y = 0;
				Mat block;
				//int bins = pow((double)2, (double)P);
				int bins = (P*(P - 1) + 3);
				cv::Mat feature_hist(1, ((row_blocks*col_blocks + 1))*(bins), CV_32FC1, Scalar(0));

				eyes_cascade.detectMultiScale(frame, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

				int prevsize = 0;
				int size = bins - 1;

				//cout << "I am in loop" << endl;

				cv::Mat eye_temp_hist(1, bins, CV_32FC1, Scalar(0));
				cv::Mat block_temp_hist(1, bins, CV_32FC1, Scalar(0));


				for (size_t j = 0; j < eyes.size(); j++)
				{
					int kk = eyes.size() - 1;
					if (kk == eyes.size() - 1)
					{
						//rectangle(frame, Point(eyes[j].x, eyes[j].y), Point(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height), Scalar(255, 0, 0), 1, 8, 0);
						Mat eye_temp;

						eye_temp = frame(eyes.at(j));

						eye_temp_hist = getLBPu2Hist(eye_temp, P, R);
						normalizeHist(eye_temp_hist);
						eye_temp_hist.colRange(0, size).copyTo(feature_hist.colRange(prevsize, prevsize + size));

					}
				}

				for (int i = 0; i < row_blocks; i++)
				{
					for (int j = 0; j < col_blocks; j++)
					{
						Rect roi = Rect(init_x, init_y, size_of_row_blocks - 1, size_of_col_blocks - 1);
						block = frame(roi);
						//rectangle(frame, Point(init_x, init_y), Point(init_x + size_of_row_blocks, init_y + size_of_col_blocks), Scalar(255, 255, 0), 1, 8, 0);
						/*imshow("frame", frame);
						waitKey(0);*/
						block_temp_hist = getLBPu2Hist(block, P, R);
						normalizeHist(block_temp_hist);
						prevsize = prevsize + size + 1;
						block_temp_hist.colRange(0, size).copyTo(feature_hist.colRange(prevsize, prevsize + size));

						init_x = init_x + size_of_col_blocks;

					}
					init_x = 0;
					init_y = init_y + size_of_row_blocks;
				}

				int temp_exp_index;
				temp_exp_index = prediction(feature_hist);
				output_expression = names[temp_exp_index];
				/*if (temp_exp_index == g)
				{
				score = 1;

				}
				else
				{
				score = 0;
				}
				if (ptr.is_open())
				{
				ptr << g << "\t" << temp_exp_index << "\t" << score << "\r\n";
				ptr.flush();
				}*/
				Rect roi = Rect(30, 50, frame.cols, frame.rows);
				// cv::putText(frame, output_expression, roi.tl(), CV_FONT_HERSHEY_TRIPLEX, 0.7, Scalar(0, 0, 255), 1);
				//LIVE STREAM
				cv::putText(frame_stream, output_expression, roi.tl(), CV_FONT_HERSHEY_TRIPLEX, 0.7, Scalar(0, 0, 255), 1);
				//
				/*imshow("image", frame);
				waitKey(0);*/
			}

			imshow("image", frame_stream);
			char c = (char)waitKey(33);
			if (c == 27) break;

		}
		//}


		ptr.close();
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
		cout << e.what() << endl;
		getchar();
	}
	system("pause");
	return 0;

}


int Box::prediction(const cv::Mat &testingData)
{
	try
	{

		float p;
		/*Ptr<SVM> svm = SVM::create();
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::LINEAR);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

		svm->load("SVM_LBP.xml");*/
		Ptr<SVM> svm = Algorithm::load<SVM>("SVM_LBP.xml");
		p = svm->predict(testingData);

		return(p);
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
}
