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

Tensor& Tensor::reshape(const vector<int>& shape)
{
	if (calculateTotal(shape) != total)
		throw runtime_error("Sizes do not match!");

	this->shape = shape;
	this->stride = calculateStride(shape);

	return *this;
}

__global__ void resizeKernel(float* X, float* A, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		X[idx] = A[idx];
	}
}

Tensor& Tensor::resize(const vector<int>& shape)
{
	Tensor X = zeros(shape);

	int block = 256;
	
	int copySize = (total <= X.total) ? total : X.total;
	int grid = (copySize + block - 1) / block;
	resizeKernel << <grid, block >> > (X.rawData(), data, copySize);

	*this = move(X);

	return *this;
}

Tensor& Tensor::squeeze(int dim)
{
	vector<int> new_shape;

	if (dim<-1 || dim>static_cast<int>(shape.size())-1)
		throw runtime_error("Dim value must be between -1 and tensor dimension - 1!");

	if (dim != -1)
	{
		if (shape[dim] != 1)
		{
			return *this;
		}
		for (int i = 0; i < shape.size(); i++)
		{
			if (i != dim)
				new_shape.push_back(shape[i]);
		}
	}
	else
	{
		for (int d : shape)
		{
			if (d != 1)
				new_shape.push_back(d);
		}
	}

	shape = new_shape;
	stride = calculateStride(new_shape);

	return *this;
}

Tensor& Tensor::unsqueeze(int dim)
{
	if (dim<-1 || dim>static_cast<int>(shape.size()))
		throw runtime_error("Dim value must be between -1 and tensor dimension!");

	vector<int> new_shape;

	if (dim == -1 || dim==shape.size())
	{
		new_shape = shape;
		new_shape.push_back(1);
		shape = new_shape;
		stride = calculateStride(new_shape);
		return *this;
	}
	for (int i = 0; i < shape.size(); i++)
	{
		if (i == dim)
			new_shape.push_back(1);
		new_shape.push_back(shape[i]);
	}
	shape = new_shape;
	stride = calculateStride(new_shape);
	return *this;
}

__global__ void mulKernel(const float* A, const float* B, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] * B[idx];
	}
}

Tensor Tensor::operator*(const Tensor& B) const
{
	if (shape != B.shape)
		throw runtime_error("Shapes do not match!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	mulKernel << <grid, block >> > (data, B.rawData(), C.rawData(), total);

	return C;
}

__global__ void divKernel(const float* A, const float* B, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] / B[idx];
	}
}

Tensor Tensor::operator/(const Tensor& B) const
{
	if (shape != B.shape)
		throw runtime_error("Shapes do not match!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	divKernel << <grid, block >> > (data, B.rawData(), C.rawData(), total);

	return C;
}

__global__ void addKernel(const float* A, const float* B, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] + B[idx];
	}
}

Tensor Tensor::operator+(const Tensor& B) const
{
	if (shape != B.shape)
		throw runtime_error("Shapes do not match!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	addKernel << <grid, block >> > (data, B.rawData(), C.rawData(), total);

	return C;
}

__global__ void subKernel(const float* A, const float* B, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] - B[idx];
	}
}

Tensor Tensor::operator-(const Tensor& B) const
{
	if (shape != B.shape)
		throw runtime_error("Shapes do not match!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	subKernel << <grid, block >> > (data, B.rawData(), C.rawData(), total);

	return C;
}

__global__ void mulScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] * x;
	}
}

Tensor Tensor::operator*(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	mulScalarKernel << <grid, block >> > (data, C.rawData(), x, total);

	return C;
}

__global__ void divScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] / x;
	}
}

Tensor Tensor::operator/(float x) const
{
	if (x == 0)
		throw runtime_error("ivision by zero not allowed!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	divScalarKernel << <grid, block >> > (data, C.rawData(), x, total);

	return C;
}

__global__ void addScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] + x;
	}
}

Tensor Tensor::operator+(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	addScalarKernel << <grid, block >> > (data, C.rawData(), x, total);

	return C;
}

__global__ void subScalarKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] - x;
	}
}

Tensor Tensor::operator-(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	subScalarKernel << <grid, block >> > (data, C.rawData(), x, total);

	return C;
}

Tensor operator*(float x, const Tensor& A)
{
	return A * x;
}

Tensor operator+(float x, const Tensor& A)
{
	return A + x;
}

__global__ void scalarDivKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = x / A[idx];
	}
}

Tensor operator/(float x, const Tensor& A)
{
	Tensor C(A.getShape());

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	scalarDivKernel << <grid, block >> > (A.rawData(), C.rawData(), x, A.size());

	return C;
}

__global__ void scalarSubKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = x - A[idx];
	}
}

Tensor operator-(float x, const Tensor& A)
{
	Tensor C(A.getShape());

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	scalarSubKernel << <grid, block >> > (A.rawData(), C.rawData(), x, A.size());

	return C;
}