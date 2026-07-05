#pragma once

#include <cuda_runtime.h>
#include <vector>

using namespace std;

class Tensor
{
private:
	float* data = nullptr;
	vector<int> shape;
	vector<int> stride;
	int total = 0;

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

	int size() const;

	int dim() const;
};