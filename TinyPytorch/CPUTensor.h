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
	void recursivePrint(int dim, int offset, int indent) const;

public:
	CPUTensor();

	CPUTensor(const vector<int>& shape);

	float& operator()(int index);

	const float& operator()(int index) const;

	float* rawData();

	const float* rawData() const;

	CPUTensor& operator=(const vector<float>& X);

	int size() const;

	int dim() const;

	const vector<int>& getShape() const;

	const vector<int>& getStride() const;

	Tensor toCUDA() const;

	void print() const;

	static void setSeed(int seed);

	static CPUTensor random(const vector<int>& shape);

	static CPUTensor randomUniform(const vector<int>& shape, float start, float end);

	static void recursMapping(vector<int>& I, const vector<int>& shape, const vector<int>& stride, int idx, int dim);

	static CPUTensor theMax(const CPUTensor& A, int axis);
};