#include "CPUTensor.h";
#include <stdexcept>
#include <iostream>
#include <random>

static random_device rd;
static mt19937 gen(rd());

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

void CPUTensor::recursivePrint(int dim, int offset, int indent) const
{
	int rank = static_cast<int>(shape.size());

	if (rank == 0)
	{
		cout << data[0];
		return;
	}

	if (dim == rank - 1)
	{
		cout << "[";
		for (int i = 0; i < shape[dim]; i++)
		{
			cout << data[offset + i * stride[dim]];

			if (i != shape[dim] - 1)
				cout << ", ";
		}
		cout << "]";
		return;
	}

	cout << "[";
	for (int i = 0; i < shape[dim]; i++)
	{
		int new_offset = offset + i * stride[dim];

		if (i > 0)
		{
			cout << ",\n";
			cout << string(indent + 1, ' ');
		}

		recursivePrint(dim + 1, new_offset, indent + 1);
	}
	cout << "]";
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

CPUTensor& CPUTensor::operator=(const vector<float>& X)
{
	if (X.size() != total)
		throw runtime_error("Sizes dont match!");
	data = X;

	return *this;
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

void CPUTensor::print() const
{
	recursivePrint(0, 0, 0);
	cout << "\n";
}

void CPUTensor::setSeed(int seed)
{
	gen.seed(seed);
}

CPUTensor CPUTensor::random(const vector<int>& shape)
{
	normal_distribution<float> dist(0.0f, 1.0f);

	CPUTensor T(shape);

	for (int i = 0; i < T.size(); i++)
		T(i) = dist(gen);

	return T;
}

CPUTensor CPUTensor::randomUniform(const vector<int>& shape, float start, float end)
{
	uniform_real_distribution<float> dist(start, end);

	CPUTensor T(shape);

	for (int i = 0; i < T.size(); i++)
		T(i) = dist(gen);

	return T;
}