#pragma once

#include <cuda_runtime.h>
#include <vector>

using namespace std;

class CPUTensor;

class Tensor
{
private:
	float* data = nullptr;
	vector<int> shape;
	vector<int> stride;
	int total;

	static int calculateTotal(const vector<int>& shape);
	static vector<int> calculateStride(const vector<int>& shape);

public:
	Tensor();

	Tensor(const vector<int>& shape);

	~Tensor();

	Tensor(const Tensor&) = delete;
	Tensor& operator=(const Tensor&) = delete;

	Tensor(Tensor&& other) noexcept;
	Tensor& operator=(Tensor&& other) noexcept;

	float* rawData();

	const float* rawData() const;

	Tensor& operator=(const std::vector<float>& X);

	int size() const;

	int dim() const;

	const vector<int>& getShape() const;

	const vector<int>& getStride() const;

	CPUTensor toCPU() const;
};