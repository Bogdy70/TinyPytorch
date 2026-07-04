#include "CMatrix.cuh"
#include "Matrix.h"
#include <stdexcept>
#include <random>

static std::random_device rd;
static std::mt19937 gen(rd());

CMatrix::CMatrix(): rows(0), cols(0), data(nullptr) {}

CMatrix::CMatrix(int r, int c) : rows(r), cols(c), data(nullptr)
{
	cudaMalloc(&data, r * c * sizeof(float));
}

CMatrix::~CMatrix()
{
	if(data!=nullptr)
		cudaFree(data);
}

CMatrix::CMatrix(CMatrix&& other) noexcept: rows(other.rows), cols(other.cols), data(other.data)
{
	other.rows = 0;
	other.cols = 0;
	other.data = nullptr;
}

void checkCuda(cudaError_t result, const char* message)
{
	if (result != cudaSuccess)
	{
		throw std::runtime_error(
			std::string(message) + ": " + cudaGetErrorString(result)
		);
	}
}

CMatrix& CMatrix::operator=(CMatrix&& other) noexcept
{
	if (this != &other)
	{
		if (data != nullptr)
			cudaFree(data);

		rows = other.rows;
		cols = other.cols;
		data = other.data;

		other.rows = 0;
		other.cols = 0;
		other.data = nullptr;
	}

	return *this;
}

int CMatrix::getRows() const
{
	return rows;
}

int CMatrix::getCols() const
{
	return cols;
}

int CMatrix::size() const
{
	return rows * cols;
}

float* CMatrix::rawData()
{
	return data;
}

const float* CMatrix::rawData() const
{
	return data;
}

Matrix CMatrix::toCPU() const
{
	Matrix C(rows, cols);

	cudaMemcpy(C.rawData(), data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

	return C;
}

__global__ void matmulKernel(const float* A, const float* B, float* C, int N, int M, int K)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		float sum = 0.0f;

		for (int k = 0; k < K; k++)
		{
			sum += A[i * K + k] * B[k * M + j];
		}

		C[i * M + j] = sum;
	}
}

CMatrix CMatrix::matmul(const CMatrix& B) const
{
	if (cols != B.getRows())
	{
		throw std::runtime_error(
			"Invalid shape for matrix multiplication: (" +
			std::to_string(rows) + ", " + std::to_string(cols) +
			") x (" +
			std::to_string(B.getRows()) + ", " + std::to_string(B.getCols()) +
			")"
		);
	}

	CMatrix C(rows, B.getCols());

	dim3 block(16, 16);
	dim3 grid((B.getCols() + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	matmulKernel<<<grid, block>>>(data, B.rawData(), C.rawData(), rows, B.getCols(), cols);

	checkCuda(cudaGetLastError(), "matmulKernel launch failed");
	checkCuda(cudaDeviceSynchronize(), "matmulKernel execution failed");

	return C;
}

__global__ void mulKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] * B[i * M + j];
	}
}

CMatrix CMatrix::operator*(const CMatrix& B) const
{
	if (rows != B.getRows() || cols != B.getCols())
		throw std::runtime_error("Invalid shape");

	CMatrix C(B.getRows(), B.getCols());

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	mulKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void scalarMulKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] * x;
	}
}

CMatrix CMatrix::operator*(float x) const
{
	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	scalarMulKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void divKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] / B[i * M + j];
	}
}

CMatrix CMatrix::operator/(const CMatrix& B) const
{
	if(rows!=B.getRows() || cols!=B.getCols())
		throw std::runtime_error("Invalid shape for element wise division");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	divKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void divScalarKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] / x;
	}
}

CMatrix CMatrix::operator/(float x) const
{
	if (x==0)
		throw std::runtime_error("Division by zero");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	divScalarKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void addKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] + B[i * M + j];
	}
}

CMatrix CMatrix::operator+(const CMatrix& B) const
{
	if (rows != B.getRows() || cols != B.getCols())
		throw std::runtime_error("Invalid shape for element wise addition");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	addKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void scalarAddKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] + x;
	}
}

CMatrix CMatrix::operator+(float x) const
{
	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	scalarAddKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void subKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] - B[i * M + j];
	}
}

CMatrix CMatrix::operator-(const CMatrix& B) const
{
	if (rows != B.getRows() || cols != B.getCols())
		throw std::runtime_error("Invalid shape for element wise substraction");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	subKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void subScalarKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] - x;
	}
}

CMatrix CMatrix::operator-(float x) const
{
	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	subScalarKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void equalsKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] == B[i * M + j] ? 1.0f : 0.0f;
	}
}

CMatrix CMatrix::operator==(const CMatrix& B) const
{
	if (rows != B.getRows() || cols != B.getCols())
		throw std::runtime_error("Invalid shape for equality");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	equalsKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void greaterthKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] > x ? 1.0f : 0.0f;
	}
}

CMatrix CMatrix::operator>(float x) const
{
	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	greaterthKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void lessthKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] < x ? 1.0f : 0.0f;
	}
}

CMatrix CMatrix::operator<(float x) const
{
	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	lessthKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void broadcastAddKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] + B[i];
	}
}

CMatrix CMatrix::broadcastAdd(const CMatrix& B) const
{
	if (rows != B.getRows())
		throw std::runtime_error("Invalid shape for element broadcast addition");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	broadcastAddKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void broadcastDivKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] / B[j];
	}
}

CMatrix CMatrix::broadcastDiv(const CMatrix& B) const
{
	if (cols != B.getCols() || B.getRows() != 1)
		throw std::runtime_error("Invalid shape for division with broadcast");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	broadcastDivKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void TKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[j * N + i] = A[i * M + j];
	}
}

CMatrix CMatrix::T() const
{
	CMatrix C(cols, rows);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	TKernel << <grid, block >> > (data, C.rawData(), rows, cols);

	return C;
}

CMatrix CMatrix::random(const int rows, const int cols)
{
	Matrix R = Matrix::random(rows, cols);

	return R.toCUDA();
}

CMatrix CMatrix::randomUniform(const int rows, const int cols, const float start, const float end)
{
	Matrix R = Matrix::randomUniform(rows, cols, start, end);

	return R.toCUDA();
}

__global__ void sum0Kernel(const float* A, float* C, int N, int M)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < M)
	{
		float sum = 0.0f;

		for (int i = 0; i < N; i++)
		{
			sum += A[i * M + col];
		}
		C[col] = sum;
	}
}

__global__ void sum1Kernel(const float* A, float* C, int N, int M)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N)
	{
		float sum = 0.0f;

		for (int j = 0; j < M; j++)
		{
			sum += A[row * M + j];
		}
		C[row] = sum;
	}
}

__global__ void sumallKernel(const float* A, float* C, int size)
{
	extern __shared__ float sh[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float val = 0.0f;

	if (idx < size)
		val = A[idx];

	sh[tid] = val;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride)
			sh[tid] += sh[tid + stride];

		__syncthreads();
	}

	if (tid == 0)
		atomicAdd(C, sh[0]);
}

CMatrix CMatrix::sum(const CMatrix& A, int axis)
{

	int rows = A.getRows();
	int cols = A.getCols();
	int block = 256;

	if (axis == 0)
	{
		CMatrix C(1, cols);

		int grid = (cols + block - 1) / block;

		sum0Kernel << <grid, block >> > (A.rawData(), C.rawData(), rows, cols);

		return C;
	}
	else if (axis == 1)
	{
		CMatrix C(rows, 1);

		int grid = (rows + block - 1) / block;

		sum1Kernel << <grid, block >> > (A.rawData(), C.rawData(), rows, cols);

		return C;
	}
	else if (axis == -1)
	{
		CMatrix C(1, 1);

		cudaMemset(C.rawData(), 0, sizeof(float));

		int grid = (rows * cols + block - 1) / block;
		int sharedBytes = block * sizeof(float);

		sumallKernel << <grid, block, sharedBytes >> > (A.rawData(), C.rawData(), rows*cols);

		return C;
	}
	else
		throw std::runtime_error("Invalid axis value. Please choose -1, 0 or 1");
}

__global__ void powKernel(const float* A, float* C, int N, int M, float power)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = powf(A[i * M + j], power);
	}
}

CMatrix CMatrix::powM(const CMatrix& A, float power)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	powKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols(), power);

	return C;
}

__global__ void sqrtKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = sqrtf(A[i * M + j]);
	}
}

CMatrix CMatrix::sqrtM(const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	sqrtKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols());

	return C;
}

__global__ void expKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = expf(A[i * M + j]);
	}
}

CMatrix CMatrix::expM(const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	expKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols());

	return C;
}

__global__ void logKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = logf(A[i * M + j]);
	}
}

CMatrix CMatrix::logM(const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	logKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols());

	return C;
}

__global__ void absKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i*M+j] = abs(A[i*M+j]);
	}
}

CMatrix CMatrix::absM(const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.cols + block.x - 1) / block.x, (A.rows + block.y - 1) / block.y);

	absKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols());

	return C;
}

__global__ void tanhKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = tanhf(A[i * M + j]);
	}
}

CMatrix CMatrix::tanhM(const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	tanhKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols());

	return C;
}

__global__ void reluKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] > 0.0f ? A[i * M + j] : 0.0f;
	}
}

CMatrix CMatrix::relu(const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	reluKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols());

	return C;
}

__global__ void der_reluKernel(const float* A, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] > 0.0f ? 1.0f : 0.0f;
	}
}

CMatrix CMatrix::der_relu(const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	der_reluKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols());

	return C;
}

__global__ void max0Kernel(const float* A, float* C, int N, int M)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < M)
	{
		float maxA = A[col];

		for (int i = 1; i < N; i++)
		{
			if(A[i*M+col]>maxA)
				maxA = A[i*M+col];
		}

		C[col] = maxA;
	}
}

__global__ void max1Kernel(const float* A, float* C, int N, int M)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N)
	{
		float maxA = A[row*M];

		for (int j = 1; j < M; j++)
		{
			if (A[row * M + j] > maxA)
				maxA = A[row * M + j];
		}

		C[row] = maxA;
	}
}

__global__ void maxallKernel(const float* A, float* partial, int size)
{
	extern __shared__ float sh[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float val = -FLT_MAX;

	if (idx < size)
		val = A[idx];

	sh[tid] = val;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride && sh[tid + stride]>sh[tid])
			sh[tid] = sh[tid + stride];

		__syncthreads();
	}

	if (tid == 0)
	{
		partial[blockIdx.x] = sh[0];
	}
}

CMatrix CMatrix::maxA(const CMatrix& A, int axis)
{
	int rows = A.getRows();
	int cols = A.getCols();
	int block = 256;

	if (axis == 0)
	{
		CMatrix C(1, cols);

		int grid = (cols + block - 1) / block;

		max0Kernel << <grid, block >> > (A.rawData(), C.rawData(), rows, cols);

		return C;
	}
	else if (axis == 1)
	{
		CMatrix C(rows, 1);

		int grid = (rows + block - 1) / block;

		max1Kernel << <grid, block >> > (A.rawData(), C.rawData(), rows, cols);

		return C;
	}
	else if (axis == -1)
	{
		int grid = (rows * cols + block - 1) / block;
		int sharedBytes = block * sizeof(float);

		CMatrix partial(1, grid);

		maxallKernel << <grid, block, sharedBytes >> > (A.rawData(), partial.rawData(), rows * cols);

		cudaDeviceSynchronize();

		int currentSize = grid;

		while (currentSize > 1)
		{
			int nextGrid = (currentSize + block - 1) / block;
			CMatrix next(1, nextGrid);

			maxallKernel << <nextGrid, block, sharedBytes >> > (partial.rawData(), next.rawData(), currentSize);

			cudaDeviceSynchronize();

			currentSize = nextGrid;
			partial = std::move(next);
		}

		return partial;
	}
	else
		throw std::runtime_error("Invalid axis value. Please input -1, 0 or 1");
}

__global__ void argmax0Kernel(const float* A, float* C, int N, int M)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < M)
	{
		int id = 0;
		float maxA = A[col];

		for (int i = 1; i < N; i++)
		{
			if (A[i * M + col] > maxA)
			{
				maxA = A[i * M + col];
				id = i;
			}
		}

		C[col] = id;
	}
}

__global__ void argmax1Kernel(const float* A, float* C, int N, int M)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N)
	{
		int id = 0;
		float maxA = A[row*M];

		for (int j = 1; j < M; j++)
		{
			if (A[row * M + j] > maxA)
			{
				maxA = A[row * M + j];
				id = j;
			}
		}

		C[row] = id;
	}
}

CMatrix CMatrix::argmax(const CMatrix& A, int axis)
{
	int rows = A.getRows();
	int cols = A.getCols();
	int block = 256;

	if (axis == 0)
	{
		CMatrix C(1, cols);

		int grid = (cols + block - 1) / block;

		argmax0Kernel << <grid, block >> > (A.rawData(), C.rawData(), rows, cols);

		return C;
	}
	else if (axis == 1)
	{
		CMatrix C(rows, 1);

		int grid = (rows + block - 1) / block;

		argmax1Kernel << <grid, block >> > (A.rawData(), C.rawData(), rows, cols);

		return C;
	}
	else
		throw std::runtime_error("Invalid axis value. Pleasse input 0 or 1");
}

__global__ void clipKernel(const float* A, float* C, int N, int M, float minVal, float maxVal)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		if (A[i * M + j] > minVal)
		{
			if (A[i * M + j] < maxVal)
			{
				C[i * M + j] = A[i * M + j];
			}
			else
				C[i * M + j] = maxVal;
		}
		else
			C[i * M + j] = minVal;
	}
}

CMatrix CMatrix::clipM(const CMatrix& A, float minVal, float maxVal)
{
	if (minVal > maxVal)
		throw std::runtime_error("Min value cannot be bigger than the max value");

	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	clipKernel << <grid, block >> > (A.rawData(), C.rawData(), A.getRows(), A.getCols(), minVal, maxVal);

	return C;
}

__global__ void scalarDivKernel(const float* A, float* C, float x, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = x / A[i * M + j];
	}
}

CMatrix operator/(float x, const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	scalarDivKernel << <grid, block >> > (A.rawData(), C.rawData(), x, A.getRows(), A.getCols());

	return C;
}

__global__ void scalarSubKernel(const float* A, float* C, float x, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = x - A[i * M + j];
	}
}

CMatrix operator-(float x, const CMatrix& A)
{
	CMatrix C(A.getRows(), A.getCols());

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	scalarSubKernel << <grid, block >> > (A.rawData(), C.rawData(), x, A.getRows(), A.getCols());

	return C;
}

CMatrix operator+(float x, const CMatrix& A)
{
	return A + x;
}

CMatrix operator*(float x, const CMatrix& A)
{
	return A * x;
}

CMatrix CMatrix::clone() const
{
	CMatrix C(rows, cols);

	cudaMemcpy(C.rawData(), data, size() * sizeof(float), cudaMemcpyDeviceToDevice);

	return C;
}

float CMatrix::toScalar() const
{
	if (rows != 1 || cols != 1)
		throw std::runtime_error("Invalid shape for a scalar matrix");

	float value = 0.0f;

	cudaMemcpy(&value, data, sizeof(float), cudaMemcpyDeviceToHost);

	return value;
}