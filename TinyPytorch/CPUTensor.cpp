#include "CPUTensor.h";
#include <stdexcept>

CPUTensor::CPUTensor(): data(), shape(), stride(), total(0) {}

int CPUTensor::calculateTotal(const vector<int>& shape)
{
	int res = 1;

	for (int i = 0; i < static_cast<int>(shape.size()); i++)
	{
		if (shape[i] <= 0)
			throw runtime_error("Tensor dimension cannot be less than or 0.");
		res *= shape[i];
	}

	return res;
}

vector<int> CPUTensor::calculateStride(const vector<int>& shape)
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

CPUTensor::CPUTensor(const vector<int>& shape) : data(), shape(shape), stride(calculateStride(shape)), total(0)
{
	total = calculateTotal(shape);
	data.resize(total);
}

float& CPUTensor::operator()(int index)
{
	return data[index];
}

const float& CPUTensor::operator()(int index) const
{
	return data[index];
}

float* CPUTensor::rawData()
{
	return data.data();
}

const float* CPUTensor::rawData() const
{
	return data.data();
}

int CPUTensor::size() const
{
	return total;
}

int CPUTensor::dim() const
{
	return static_cast<int>(shape.size());
}

const vector<int>& CPUTensor::getShape() const
{
	return shape;
}

const vector<int>& CPUTensor::getStride() const
{
	return stride;
}

Tensor CPUTensor::toCUDA() const
{
	Tensor T(shape);

	cudaMemcpy(T.rawData(), data.data(), total * sizeof(float), cudaMemcpyHostToDevice);

	return T;
}