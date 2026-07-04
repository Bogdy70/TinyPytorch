#pragma once

#include <cuda_runtime.h>

class Matrix;

class CMatrix
{
private:
	int rows;
	int cols;
	float* data;

public:
	CMatrix();

	CMatrix(int r, int c);

	~CMatrix();

	CMatrix(const CMatrix&) = delete;
	CMatrix& operator=(const CMatrix&) = delete;

	CMatrix(CMatrix&& other) noexcept;
	CMatrix& operator=(CMatrix&& other) noexcept;

	int getRows() const;

	int getCols() const;

	int size() const;

	float* rawData();

	const float* rawData() const;

	Matrix toCPU() const;

	CMatrix matmul(const CMatrix& B) const;

	CMatrix operator*(const CMatrix& B) const;

	CMatrix operator*(float x) const;

	CMatrix operator/(const CMatrix& B) const;

	CMatrix operator/(float x) const;

	CMatrix operator+(const CMatrix& B) const;

	CMatrix operator+(float x) const;

	CMatrix operator-(const CMatrix& B) const;

	CMatrix operator-(float x) const;

	CMatrix operator==(const CMatrix& B) const;

	CMatrix operator>(float x) const;

	CMatrix operator<(float x) const;

	CMatrix broadcastAdd(const CMatrix& B) const;

	CMatrix broadcastDiv(const CMatrix& B) const;

	CMatrix T() const;

	static CMatrix random(const int rows, const int cols);

	static CMatrix randomUniform(const int rows, const int cols, const float start, const float end);

	static CMatrix zeros(const int rows, const int cols);

	static CMatrix sum(const CMatrix& A, int axis = -1);

	static CMatrix powM(const CMatrix& A, float power);

	static CMatrix sqrtM(const CMatrix& A);

	static CMatrix expM(const CMatrix& A);

	static CMatrix logM(const CMatrix& A);

	static CMatrix absM(const CMatrix& A);

	static CMatrix tanhM(const CMatrix& A);

	static CMatrix relu(const CMatrix& A);

	static CMatrix der_relu(const CMatrix& A);

	static CMatrix maxA(const CMatrix& A, int axis = -1);

	static CMatrix argmax(const CMatrix& A, int axis = 0);

	static CMatrix clipM(const CMatrix& A, float minVal, float maxVal);

	CMatrix clone() const;

	float toScalar() const;
};

CMatrix operator/(float x, const CMatrix& A);

CMatrix operator-(float x, const CMatrix& A);

CMatrix operator+(float x, const CMatrix& A);

CMatrix operator*(float x, const CMatrix& A);