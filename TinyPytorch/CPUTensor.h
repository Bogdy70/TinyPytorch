#pragma once

#include <vector>
#include "Tensor.cuh"

using namespace std;

class CPUTensor
{
private:
	vector<float> data;
	vector<int> shape;
	vector<int> stride;
	int total;

	static int calculateTotal(const vector<int>& shape);
	static vector<int> calculateStride(const vector<int>& shape);

public:
	CPUTensor();

	CPUTensor(const vector<int>& shape);

	float& operator()(int index);

	const float& operator()(int index) const;

	float* rawData();

	const float* rawData() const;

	int size() const;

	int dim() const;

	const vector<int>& getShape() const;

	const vector<int>& getStride() const;

	Tensor toCUDA() const;
};