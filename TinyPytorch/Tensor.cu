#include "Tensor.cuh"
#include "CPUTensor.h"
#include <stdexcept>

Tensor::Tensor(): data(nullptr), shape(), stride(), total(0) {}

int Tensor::calculateTotal(const vector<int>& shape)
{
	int res = 1;
	for (int i = 0; i < static_cast<int>(shape.size()); i++)
	{
		if (shape[i] <= 0)
			throw runtime_error("Tensor dimensions must be pozitive.");
		res *= shape[i];
	}

	return res;
}

vector<int> Tensor::calculateStride(const vector<int>& shape)
{
	vector<int> res(shape.size());
	int val = 1;
	for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
	{
		res[i] = val;
		val *= shape[i];
	}

	return res;
}

Tensor::Tensor(const vector<int>& shape): data(nullptr), shape(shape), stride(calculateStride(shape)), total(calculateTotal(shape))
{
	cudaMalloc(&data, total * sizeof(float));
}

Tensor::~Tensor()
{
	if (data != nullptr)
		cudaFree(data);
}

Tensor::Tensor(Tensor&& other) noexcept: data(other.data), shape(move(other.shape)), stride(move(other.stride)), total(other.total)
{
	other.data = nullptr;
	other.shape = {};
	other.stride = {};
	other.total = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
	if (this != &other)
	{
		if (data != nullptr)
			cudaFree(data);

		data = other.data;
		shape = move(other.shape);
		stride = move(other.stride);
		total = other.total;

		other.data = nullptr;
		other.shape = {};
		other.stride = {};
		other.total = 0;
	}

	return *this;
}

float* Tensor::rawData()
{
	return data;
}

const float* Tensor::rawData() const
{
	return data;
}

Tensor& Tensor::operator=(const vector<float>& X)
{
	if (X.size() != total)
		throw runtime_error("Sizes do not match!");

	cudaMemcpy(data, X.data(), total * sizeof(float), cudaMemcpyHostToDevice);

	return *this;
}

int Tensor::size() const
{
	return total;
}

int Tensor::dim() const
{
	return static_cast<int>(shape.size());
}

const vector<int>& Tensor::getShape() const
{
	return shape;
}

const vector<int>& Tensor::getStride() const
{
	return stride;
}

CPUTensor Tensor::toCPU() const
{
	CPUTensor T(shape);

	cudaMemcpy(T.rawData(), data, total * sizeof(float), cudaMemcpyDeviceToHost);

	return T;
}

Tensor Tensor::zeros(const vector<int>& shape)
{
	Tensor T(shape);
	cudaMemset(T.data, 0, T.size() * sizeof(float));
	return T;
}

Tensor Tensor::random(const vector<int>& shape)
{
	return CPUTensor::random(shape).toCUDA();
}

Tensor Tensor::randomUniform(const vector<int>& shape, float start, float end)
{
	return CPUTensor::randomUniform(shape, start, end).toCUDA();
}

__global__ void fillKernel(float* T, int size, float value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		T[idx] = value;
	}
}

Tensor Tensor::fill(const vector<int>& shape, float value)
{
	Tensor T(shape);

	int block = 256;
	int grid = (T.size() + block - 1) / block;

	fillKernel << <grid, block >> > (T.rawData(), T.size(), value);

	return T;
}