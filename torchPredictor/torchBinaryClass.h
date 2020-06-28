#pragma once
#include <vector>
#include <inttypes.h>

class __declspec(dllexport) TorchBinaryClass
//class TorchBinaryClass
{
public:
	enum DESTINATION {CPU, CUDA};

	void Init(const char *filePath, DESTINATION onCPU = CPU);

	void compute(uint8_t *dataIn, float *dataOut, int rows, int cols);

	TorchBinaryClass();
	~TorchBinaryClass();
};

//__declspec(dllexport) TorchBinaryClass *Create(const std::string &filePath, TorchBinaryClass::DESTINATION onCPU)
//{
//	return new TorchBinaryClass(filePath, onCPU);
//}
//
