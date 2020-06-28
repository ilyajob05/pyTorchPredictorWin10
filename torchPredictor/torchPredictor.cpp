#include "torchPredictor.h"
#include "torchBinaryClass.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>



using namespace std;

int main(void)
{
	TorchBinaryClass eyePredictor;
	//string path("C:/projects/pyTorch/binaryClassification/torch_model_eval2019-02-16__17_16epoch_23__acc_88.298.pthcpp"); // zbs
	string path("C:/projects/pyTorch/binaryClassification/torch_model_eval2019-02-16__19_52epoch_500__acc_94.563.pthcpp"); // zbs

	eyePredictor.Init(path.c_str(), TorchBinaryClass::CPU);

	vector<uint> stateEyeBuff;
	float resNet[2];

	cv::Mat inImg;

	// open
	inImg = cv::imread("C:/DATASETS/dataset_B_Eye_Images/test/openEyes/Robbie_Naish_0001_R.jpg");
	cv::cvtColor(inImg, inImg, cv::COLOR_BGR2GRAY);
	cv::resize(inImg, inImg, cv::Size(24, 24), 0.0, 0.0, cv::INTER_LANCZOS4);
	inImg = inImg.reshape(1, 1);
	eyePredictor.compute(inImg.data, resNet, 24, 24);
	cout << resNet[0] << " " << resNet[1] << "\n  result:   \t" << (resNet[1] < resNet[0]) << endl;


	// close
	inImg = cv::imread("C:/DATASETS/dataset_B_Eye_Images/test/closeEyes/closed_eye_2442.jpg_face_1_L.jpg");
	cv::cvtColor(inImg, inImg, cv::COLOR_BGR2GRAY);
	cv::resize(inImg, inImg, cv::Size(24, 24), 0.0, 0.0, cv::INTER_LANCZOS4);
	inImg = inImg.reshape(1, 1);
	eyePredictor.compute(inImg.data, resNet, 24, 24);
	cout << resNet[0] << " " << resNet[1] << "\n  result:   \t" << (resNet[1] < resNet[0]) << endl;


	// open
	inImg = cv::imread("C:/projects/QtGuiApplication1/QtGuiApplication1/0_17.jpg");
	cv::cvtColor(inImg, inImg, cv::COLOR_BGR2GRAY);
	cv::resize(inImg, inImg, cv::Size(24, 24), 0.0, 0.0, cv::INTER_LANCZOS4);
	inImg = inImg.reshape(1, 1);
	eyePredictor.compute(inImg.data, resNet, 24, 24);
	cout << resNet[0] << " " << resNet[1] << "\n  result:   \t" << (resNet[1] < resNet[0]) << endl;


	// close
	inImg = cv::imread("C:/projects/QtGuiApplication1/QtGuiApplication1/0_34.jpg");
	cv::cvtColor(inImg, inImg, cv::COLOR_BGR2GRAY);
	cv::resize(inImg, inImg, cv::Size(24, 24), 0.0, 0.0, cv::INTER_LANCZOS4);
	inImg = inImg.reshape(1, 1);
	eyePredictor.compute(inImg.data, resNet, 24, 24);
	cout << resNet[0] << " " << resNet[1] << "\n  result:   \t" << (resNet[1] < resNet[0]) << endl;


	// open
	inImg = cv::imread("C:/DATASETS/mrlEyes_2018_01/s0019/s0019_06113_0_0_1_0_0_01.png");
	cv::cvtColor(inImg, inImg, cv::COLOR_BGR2GRAY);
	cv::resize(inImg, inImg, cv::Size(24, 24), 0.0, 0.0, cv::INTER_LANCZOS4);
	inImg = inImg.reshape(1, 1);
	eyePredictor.compute(inImg.data, resNet, 24, 24);
	cout << resNet[0] << " " << resNet[1] << "\n  result:   \t" << (resNet[1] < resNet[0]) << endl;
	

	// close
	inImg = cv::imread("C:/DATASETS/mrlEyes_2018_01/s0019/s0019_00003_0_0_0_0_0_01.png");
	cv::cvtColor(inImg, inImg, cv::COLOR_BGR2GRAY);
	cv::resize(inImg, inImg, cv::Size(24, 24), 0.0, 0.0, cv::INTER_LANCZOS4);
	inImg = inImg.reshape(1, 1);
	eyePredictor.compute(inImg.data, resNet, 24, 24);
	cout << resNet[0] << " " << resNet[1] << "\n  result:   \t" << (resNet[1] < resNet[0]) << endl;

	

	string s;
	cin >> s;

	return 100;
}