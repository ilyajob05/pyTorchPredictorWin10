#include "torchBinaryClass.h"
#include "torchPredictorRaw.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <windows.h>
#include <consoleapi.h>


using namespace std;

TorchPredictor *predictor;

TorchBinaryClass::TorchBinaryClass()
{
	;
}

void TorchBinaryClass::Init(const char *filePath, DESTINATION onCPU)
{
	//AllocConsole();
	//freopen("CONOUT$", "w", stdout);
	//freopen("CONOUT$", "w", stderr);

	if (onCPU == CPU)
	{
		predictor = new TorchPredictor(TorchPredictor::CPU);
	}
	else
	{
		predictor = new TorchPredictor(TorchPredictor::CUDA);
	}

	predictor->loadConfiguration(std::string(filePath));
}


TorchBinaryClass::~TorchBinaryClass()
{
	delete predictor;
}
 

void TorchBinaryClass::compute(uint8_t *dataIn, float *dataOut, int rows, int cols)
{
	vector<uint8_t> dataInV(dataIn, dataIn + rows * cols);
	vector<float> res = predictor->compute(dataInV, rows, cols);
	memcpy(dataOut, &res[0], sizeof(float)*2);

}


