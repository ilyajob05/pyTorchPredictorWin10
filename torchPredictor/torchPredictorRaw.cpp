#include "torchPredictorRaw.h"
#include <iostream>
#include <string>


using namespace std;

TorchPredictor::TorchPredictor(TorchPredictor::DESTINATION onCPU)
{
	devDestination = (onCPU == CPU) ? at::kCPU : at::kCUDA;
}


TorchPredictor::~TorchPredictor()
{
}
 


vector<float> TorchPredictor::compute(std::vector<uint8_t> dataIn, int rows, int cols)
{
	vector<float> result;
	std::vector<int64_t> sizes = { 1, 1, rows, cols };
	at::TensorOptions options(at::ScalarType::Byte);
	at::Tensor tensor_image = torch::from_blob(&dataIn.front(), at::IntList(sizes), options);

	tensor_image = tensor_image.toType(at::kFloat);

//	std::vector<torch::jit::IValue> inputs;
//	inputs.push_back(torch::ones({ 1, 1, 24, 24 }).to(devDestination));
//	auto output = net->forward(inputs);

	tensor_image.div_(255.0f);
	tensor_image.sub_(tensor_image.min());
	tensor_image.div_(tensor_image.max() - tensor_image.min());

	// debug
	//tensor_image.sub_(tensor_image.mean());
	//tensor_image.div_(tensor_image.std());
	//cout << "mean: " << tensor_image.mean() << endl;
	//cout << "min: " << tensor_image.min() << endl;
	//cout << "max: " << tensor_image.max() << endl;
	//cout << "std: " << tensor_image.std() << endl << endl;
	//cout << "tensor: " << tensor_image << endl << endl;

	tensor_image = tensor_image.to(devDestination);
	auto output = net->forward({tensor_image});
	at::Tensor out2 = output.toTensor().to(at::kCPU);
	float data = out2.slice(/*dim=*/1, /*start=*/0, /*end=*/2)[0][0].item<float>();
	result.push_back(data);
	data = out2.slice(/*dim=*/1, /*start=*/0, /*end=*/2)[0][1].item<float>();
	result.push_back(data);
	return result;
}


vector<std::vector<float>> TorchPredictor::compute(std::vector<std::vector<float>> dataIn)
{
	vector<vector<float>> result;
	return result;
}


vector<int> TorchPredictor::applyThreshold(std::vector<std::vector<float>>)
{
	vector<int> result;
	return result;
}


int TorchPredictor::applyThreshold(std::vector<float>)
{
	int result;
	return result;
}


bool TorchPredictor::loadConfiguration(std::string filePath)
{
	bool result = true;
	net = torch::jit::load(filePath, devDestination);

	return result;
}

