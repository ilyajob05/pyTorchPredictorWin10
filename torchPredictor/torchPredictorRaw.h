#pragma once
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <inttypes.h>


class TorchPredictor
{
public:
	enum DESTINATION {CPU, CUDA};
	float thrasholdOfROC = { 0.0f };
	float probability = { 0.0f };

	std::shared_ptr<torch::jit::script::Module> net;

	TorchPredictor(DESTINATION onCPU = CPU);
	~TorchPredictor();

	bool loadConfiguration(std::string filePath);
	std::vector<float> compute(std::vector<uint8_t> dataIn, int rows, int cols);
	std::vector<std::vector<float>> compute(std::vector<std::vector<float>> dataIn);

	// get number of multiclass array
	std::vector<int> applyThreshold(std::vector<std::vector<float>>);
	// get number of multiclass
	int applyThreshold(std::vector<float>);

private:
	c10::DeviceType devDestination;
};

