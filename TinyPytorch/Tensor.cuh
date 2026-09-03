#pragma once

#include <cuda_runtime.h>
#include <vector>

using namespace std;

class CPUTensor;

struct MaxPoolRes;

struct MaxStats
{
	float value;
	int index;
};

constexpr int MAX_TENSOR_LENGTH = 8;

struct BroadcastStats
{
	int out_shape[MAX_TENSOR_LENGTH]{};
	int out_stride[MAX_TENSOR_LENGTH]{};
	int strideA[MAX_TENSOR_LENGTH]{};
	int strideB[MAX_TENSOR_LENGTH]{};
};

class Tensor
{
private:
	float* data = nullptr;
	vector<int> shape;
	vector<int> stride;
	int total;

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

	Tensor& operator=(const std::vector<float>& X);

	int size() const;

	int dim() const;

	const vector<int>& getShape() const;

	const vector<int>& getStride() const;

	CPUTensor toCPU() const;

	static Tensor zeros(const vector<int>& shape);

	static Tensor fill(const vector<int>& shape, float value);

	static Tensor random(const vector<int>& shape);

	static Tensor randomUniform(const vector<int>& shape, float start, float end);

	Tensor& reshape(const vector<int>& shape);

	Tensor& resize(const vector<int>& shape);

	Tensor& squeeze(int dim = -1);

	Tensor& unsqueeze(int dim = 0);

	Tensor& flatten(int start_dim = 0, int end_dim = -1);

	static Tensor pad(const Tensor& A, int padding, float val = 0.0f);

	static Tensor conv2D(const Tensor& A, const Tensor& K, int kernel_size = 3, int hStride = 1, int vStride = 1, int padding = 0);

	static MaxPoolRes maxPool2D(const Tensor& A, int kernel_size = 2, int hStride = 1, int vStride = 1, int padding = 0);

	Tensor operator*(const Tensor& B) const;

	Tensor operator/(const Tensor& B) const;

	Tensor operator+(const Tensor& B) const;

	Tensor operator-(const Tensor& B) const;

	Tensor operator*(float x) const;

	Tensor operator/(float x) const;

	Tensor operator+(float x) const;

	Tensor operator-(float x) const;

	Tensor operator==(const Tensor& B) const;

	Tensor operator>(float x) const;

	Tensor operator<(float x) const;

	Tensor matmul(const Tensor& B) const;

	Tensor T() const;

	static Tensor sum(const Tensor& A, int axis = -1, bool keepdim = false);

	static Tensor maxT(const Tensor& A, int axis = -1, bool keepdim = false);

	static Tensor argmax(const Tensor& A, int axis = -1, bool keepdim = false);

	static Tensor powT(const Tensor& A, float power);

	static Tensor sqrtT(const Tensor& A);

	static Tensor expT(const Tensor& A);

	static Tensor logT(const Tensor& A);

	static Tensor absT(const Tensor& A);

	static Tensor clipT(const Tensor& A, float minVal, float maxVal);

	static Tensor tanhT(const Tensor& A);

	static Tensor relu(const Tensor& A);

	static Tensor der_relu(const Tensor& A);

	Tensor clone() const;

	float toScalar() const;
};

Tensor operator*(float x, const Tensor& A);

Tensor operator+(float x, const Tensor& A);

Tensor operator/(float x, const Tensor& A);

Tensor operator-(float x, const Tensor& A);

struct MaxPoolRes
{
	Tensor vals;
	Tensor idxs;

	MaxPoolRes(vector<int>& shape): vals(shape), idxs(shape) {}
};