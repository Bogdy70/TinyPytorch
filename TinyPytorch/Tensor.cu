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

__global__ void equalKernel(const float* A, const float* B, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = (A[idx] == B[idx]);
	}
}

Tensor Tensor::operator==(const Tensor& B) const
{
	if (shape != B.shape)
		throw runtime_error("Shapes do not match!");

	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	equalKernel << <grid, block >> > (data, B.rawData(), C.rawData(), total);

	return C;
}

__global__ void greaterthKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = (A[idx] > x);
	}
}

Tensor Tensor::operator>(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	greaterthKernel << <grid, block >> > (data, C.rawData(), x, total);

	return C;
}

__global__ void lessthKernel(const float* A, float* C, float x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = (A[idx] < x);
	}
}

Tensor Tensor::operator<(float x) const
{
	Tensor C(shape);

	int block = 256;
	int grid = (total + block - 1) / block;

	lessthKernel << <grid, block >> > (data, C.rawData(), x, total);

	return C;
}

__global__ void matmulKernel(float* C, const float* A, const float* B, int N, int M, int K)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<N && j<M)
	{
		float sum = 0.0f;

		for (int k = 0; k < K; k++)
		{
			sum += A[i * K + k] * B[k * M + j];
		}
		C[i * M + j] = sum;
	}
}

Tensor Tensor::matmul(const Tensor& B) const
{
	if (shape.size() > 2 || B.dim() > 2)
		throw runtime_error("Tensor must be 2 dimensional!");

	if (shape[1] != B.shape[0])
		throw runtime_error("Inner dimesnions must be equal!");

	Tensor C({ shape[0], B.shape[1] });

	dim3 block(16, 16);
	dim3 grid((B.shape[1] + block.x - 1) / block.x, (shape[0] + block.y - 1) / block.y);

	matmulKernel << <grid, block >> > (C.data, data, B.data, shape[0], B.shape[1], shape[1]);

	return C;
}

__global__ void TKernel(float* C, const float* A, int N, int M)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[j * N + i] = A[i * M + j];
	}
}

Tensor Tensor::T() const
{
	if (shape.size() > 2)
		throw runtime_error("Tensor must be 2 dimensional!");

	Tensor C({ shape[1], shape[0] });

	dim3 block(16, 16);
	dim3 grid((shape[1] + block.x - 1) / block.x, (shape[0] + block.y - 1) / block.y);

	TKernel << <grid, block >> > (C.data, data, shape[0], shape[1]);

	return C;
}

__global__ void brcstaddKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] + B[i];
	}
}

Tensor Tensor::broadcastAdd(const Tensor& B) const
{
	if (shape.size() > 2 || B.dim() > 2)
		throw runtime_error("Tensors must be 2 dimensional!");

	if (shape[0] != B.shape[0])
		throw runtime_error("Dimensions are not identical!");

	Tensor C(shape);

	dim3 block(16, 16);
	dim3 grid((shape[1] + block.x - 1) / block.x, (shape[0] + block.y - 1) / block.y);

	brcstaddKernel << <grid, block >> > (data, B.data, C.data, shape[0], shape[1]);

	return C;
}

__global__ void brdcstdivKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] / B[j];
	}
}

Tensor Tensor::broadcastDiv(const Tensor& B) const
{
	if (shape.size() > 2 || B.dim() > 2)
		throw runtime_error("Tensors must be 2 dimensional!");

	if (shape[1] != B.shape[1])
		throw runtime_error("Dimensions do not match!");

	Tensor C(shape);

	dim3 block(16, 16);
	dim3 grid((shape[1] + block.x - 1) / block.x, (shape[0] + block.y - 1) / block.y);

	brdcstdivKernel << <grid, block >> > (data, B.data, C.data, shape[0], shape[1]);

	return C;
}

__global__ void sum0Kernel(float* C, const float* A, int N, int M)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < M)
	{
		float sum = 0.0f;
		for (int i = 0; i < N; i++)
		{
			sum += A[i * M + j];
		}
		C[j] = sum;
	}
}

__global__ void sum1Kernel(float* C, const float* A, int N, int M)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		float sum = 0.0f;
		for (int j = 0; j < M; j++)
		{
			sum += A[i * M + j];
		}
		C[i] = sum;
	}
}

__global__ void sumallKernel(float* C, const float* A, int size)
{
	extern __shared__ float sh[];

	int sh_idx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float val = 0.0f;

	if (idx < size)
		val = A[idx];

	sh[sh_idx] = val;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (sh_idx < stride)
			sh[sh_idx] += sh[sh_idx + stride];

		__syncthreads();
	}

	if (sh_idx == 0)
		atomicAdd(C, sh[0]);
}

Tensor Tensor::sum(const Tensor& A, int axis)
{
	if (A.dim() > 2)
		throw runtime_error("Tensor must be 2 dimensional!");

	int block = 256;

	if (axis == 0)
	{
		Tensor C({ 1, A.shape[1] });

		int grid = (A.shape[1] + block - 1) / block;

		sum0Kernel << <grid, block >> > (C.data, A.data, A.shape[0], A.shape[1]);

		return C;
	}
	else if (axis == 1)
	{
		Tensor C({ A.shape[0], 1 });

		int grid = (A.shape[0] + block - 1) / block;

		sum1Kernel << <grid, block >> > (C.data, A.data, A.shape[0], A.shape[1]);

		return C;
	}
	else if (axis == -1)
	{
		Tensor C({ 1 });

		int grid = (A.total + block - 1) / block;

		cudaMemset(C.data, 0, sizeof(float));

		sumallKernel << <grid, block, block * sizeof(float) >> > (C.data, A.data, A.total);

		return C;
	}
	else
		throw runtime_error("Invalid axis value. Please input -1, 0 or 1!");
}

__global__ void argmax0Kernel(float* C, const float* A, int N, int M)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (j < M)
	{
		float max0 = A[j];
		int idx = 0;
		for (int i = 0; i < N; i++)
		{
			if (A[i * M + j] > max0)
			{
				max0 = A[i * M + j];
				idx = i;
			}
		}
		C[j] = idx;
	}
}

__global__ void argmax1Kernel(float* C, const float* A, int N, int M)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		float max1 = A[i * M];
		int idx = 0;
		for (int j = 0; j < M; j++)
		{
			if (A[i * M + j] > max1)
			{
				max1 = A[i * M + j];
				idx = j;
			}
		}
		C[i] = idx;
	}
}

Tensor Tensor::argmax(const Tensor& A, int axis)
{
	if (A.dim() > 2)
		throw runtime_error("Tensor must be 2 dimensional!");

	int block = 256;

	if (axis == 0)
	{
		Tensor C({ 1, A.shape[1] });

		int grid = (A.shape[1] + block - 1) / block;

		argmax0Kernel << <grid, block >> > (C.data, A.data, A.shape[0], A.shape[1]);

		return C;
	}
	else if (axis == 1)
	{
		Tensor C({ A.shape[0], 1 });

		int grid = (A.shape[0] + block - 1) / block;

		argmax1Kernel << <grid, block >> > (C.data, A.data, A.shape[0], A.shape[1]);

		return C;
	}
	else
		throw runtime_error("Inavlid shape. Please input 0 or 1!");
}

__global__ void powKernel(const float* A, float* C, float p, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = powf(A[idx], p);
	}
}

Tensor Tensor::powT(const Tensor& A, float power)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	powKernel << <grid, block >> > (A.data, C.data, power, A.total);

	return C;
}

__global__ void sqrtKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = sqrt(A[idx]);
	}
}

Tensor Tensor::sqrtT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	sqrtKernel << <grid, block >> > (A.data, C.data, A.total);

	return C;
}

__global__ void expKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = exp(A[idx]);
	}
}

Tensor Tensor::expT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	expKernel << <grid, block >> > (A.data, C.data, A.total);

	return C;
}

__global__ void logKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = log(A[idx]);
	}
}

Tensor Tensor::logT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	logKernel << <grid, block >> > (A.data, C.data, A.total);

	return C;
}

__global__ void absKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = abs(A[idx]);
	}
}

Tensor Tensor::absT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	absKernel << <grid, block >> > (A.data, C.data, A.total);

	return C;
}

__global__ void clipKernel(const float* A, float* C, float minVal, float maxVal, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if (A[idx] < maxVal)
		{
			if (A[idx] > minVal)
			{
				C[idx] = A[idx];
			}
			else
				C[idx] = minVal;
		}
		else
			C[idx] = maxVal;
	}
}

Tensor Tensor::clipT(const Tensor& A, float minVal, float maxVal)
{
	if (minVal > maxVal)
		throw runtime_error("Min value cannot be bigger than max value!");

	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	clipKernel << <grid, block >> > (A.data, C.data, minVal, maxVal, A.total);

	return C;
}

__global__ void tanhKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = tanh(A[idx]);
	}
}

Tensor Tensor::tanhT(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.total + block - 1) / block;

	tanhKernel << <grid, block >> > (A.data, C.data, A.total);

	return C;
}

__global__ void reluKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] > 0.0f ? A[idx] : 0.0f;
	}
}

Tensor Tensor::relu(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	reluKernel << <grid, block >> > (A.data, C.data, A.total);

	return C;
}

__global__ void der_reluKernel(const float* A, float* C, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		C[idx] = A[idx] > 0.0f ? 1.0f : 0.0f;
	}
}

Tensor Tensor::der_relu(const Tensor& A)
{
	Tensor C(A.shape);

	int block = 256;
	int grid = (A.size() + block - 1) / block;

	der_reluKernel << <grid, block >> > (A.data, C.data, A.total);

	return C;
}

Tensor Tensor::clone() const
{
	Tensor C(shape);

	cudaMemcpy(C.data, data, total * sizeof(float), cudaMemcpyDeviceToDevice);

	return C;
}

float Tensor::toScalar() const
{
	if (total != 1)
		throw runtime_error("Tensor must contain a single element!");

	float res = 0.0f;

	cudaMemcpy(&res, data, total * sizeof(float), cudaMemcpyDeviceToHost);

	return res;
}