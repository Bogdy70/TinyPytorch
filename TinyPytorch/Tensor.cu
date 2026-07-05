#include "Tensor.cuh"
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

int Tensor::size() const
{
	return total;
}

int Tensor::dim() const
{
	return static_cast<int>(shape.size());
}