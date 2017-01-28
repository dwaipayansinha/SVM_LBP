#include "global.h"
#include <fstream>
#include <iostream>
#include <stdio.h>

int Box::test()
{
	Box b;
	max1 = 0.0;
	int R = 1;
	int P = 8;
	Mat frame, frame_stream1, frame_stream5, frame_stream_gray, frame_stream, frame_new;
	int tot_score = 0, counter = 0;

	std::string output_expression;

	std::vector<std::string> names = { "Anger", "Disgust", "Fear", "Joy", "Sad", "Surprise" };
	std::vector<std::string> path{ "../CK6/Anger/", "../CK6/Disgust/", "../CK6/Fear/", "../CK6/Joy/", "../CK6/Sad", "../CK6/Surprise" };

	face_cascade_name = "../haarcascade_frontalface_alt.xml";

	trainingData = NULL;
	trainingLabel = NULL;

	Size size1(270, 270);
	int row_blocks = 9, col_blocks = 9;


	try
	{
		if (!face_cascade.load(face_cascade_name))
		{
			throw std::invalid_argument("Error Loading haarcascade_frontalface_alt.xml");
		}



		VideoCapture vcap(0);
		if (!vcap.isOpened())
		{
			throw std::invalid_argument("Error loading camera");
		}


		int e = 0, old_e_1 = 0, old_e_5 = 0, flag = 0;
		while (++e)
		{
			//cout << e << endl;
			flag = 0;

			//LIVE STREAM
			vcap >> frame_stream;
			if (frame_stream.empty())
			{
				cout << "Error : Image cannot be loaded..!!" << endl;
				return -1;
			}


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

				resize(frame, frame, size1);
				if (e == 1 || e == old_e_1 + 5)
				{
					old_e_1 = e;
					frame_stream1 = frame;

				}
				if (e == 5 || e == old_e_5 + 5)
				{
					old_e_5 = e;
					frame_stream5 = frame;
					flag = 1;
				}

			}

			if (flag == 1)
			{
				cv::absdiff(frame_stream1, frame_stream5, frame_new);
				convertScaleAbs(frame_new, frame_new);

				int size_of_row_blocks = frame_new.rows / row_blocks;
				int size_of_col_blocks = frame_new.cols / col_blocks;
				int init_x = 0, init_y = 0;
				Mat block;
				int bins = (P*(P - 1) + 3);
				cv::Mat feature_hist(1, ((row_blocks*col_blocks))*(bins), CV_32FC1, Scalar(0));



				int prevsize = 0;
				int size = bins - 1;

				cv::Mat block_temp_hist(1, bins, CV_32FC1, Scalar(0));

				for (int i = 0; i < row_blocks; i++)
				{
					for (int j = 0; j < col_blocks; j++)
					{
						Rect roi = Rect(init_x, init_y, size_of_row_blocks - 1, size_of_col_blocks - 1);
						block = frame_new(roi);
						block_temp_hist = getLBPu2Hist(block, P, R);
						normalizeHist(block_temp_hist);

						block_temp_hist.colRange(0, size).copyTo(feature_hist.colRange(prevsize, prevsize + size));
						prevsize = prevsize + size + 1;

						init_x = init_x + size_of_col_blocks;

					}
					init_x = 0;
					init_y = init_y + size_of_row_blocks;
				}

				int temp_exp_index;
				temp_exp_index = prediction(feature_hist);
				output_expression = names[temp_exp_index];
				cout << e << "   " << output_expression << endl;
				Rect roi = Rect(30, 50, frame_new.cols, frame_new.rows);
				cv::putText(frame_stream, output_expression, roi.tl(), CV_FONT_HERSHEY_TRIPLEX, 0.7, Scalar(0, 0, 255), 1);
			}

			cv::imshow("image", frame_stream);
			char c = (char)waitKey(33);
			if (c == 27) break;
		}
	}



	catch (Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
	catch (runtime_error& ex) {
		cerr << "Runtime Error:\n" << ex.what();
	}
	catch (const std::invalid_argument& e)
	{
		cout << e.what() << endl;
	}

	system("pause");
	return 0;

}


int Box::prediction(const cv::Mat &testingData)
{
	try
	{

		float p;
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
		CvSVM SVM;


		for (int j = 0; j < 6; j++)
		{
			SVM.load("SVM_LBP.xml");
			p = SVM.predict(testingData);
		}

		return(p);
	}
	catch (Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
	catch (runtime_error& ex) {
		cerr << "Runtime Error:\n" << ex.what();
	}
}
