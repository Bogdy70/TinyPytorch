#pragma once

#include <cuda_runtime.h>
#include <vector>

using namespace std;

class CPUTensor;

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

	static Tensor powM(const Tensor& A, float power);

	static Tensor sqrtM(const Tensor& A);

	static Tensor expM(const Tensor& A);

	static Tensor logM(const Tensor& A);

	static Tensor absM(const Tensor& A);

	static Tensor clipM(const Tensor& A, float minVal, float maxVal);

	static Tensor tanhM(const Tensor& A);

	static Tensor relu(const Tensor& A);

	static Tensor der_relu(const Tensor& A);

	Tensor clone() const;

	float toScalar() const;
};

Tensor operator*(float x, const Tensor& A);

Tensor operator+(float x, const Tensor& A);

Tensor operator/(float x, const Tensor& A);

Tensor operator-(float x, const Tensor& A);