#pragma once

#include<vector>
#include<string>
#include "CMatrix.cuh"

class Matrix
{
private:
    int rows;
    int cols;
    std::vector<float> data;

public:
    Matrix();

    Matrix(int r, int c);

    float& operator()(int row, int col);

    const float& operator()(int row, int col) const;

    int getRows() const;

    int getCols() const;

    float* rawData();

    const float* rawData() const;

    void setData(const std::vector<float>& X);

    int size() const;

    CMatrix toCUDA() const;

    void resize(int new_rows, int new_cols);

    void reshape(int new_rows, int new_cols);

    Matrix matmul(const Matrix& B) const;

    static Matrix loadMatrixBin(const std::string& filepath, int rows, int cols);

    Matrix operator*(const Matrix& B) const;

    Matrix operator*(float x) const;

    Matrix operator/(const Matrix& B) const;

    Matrix operator/(float x) const;

    Matrix operator+(const Matrix& B) const;

    Matrix operator+(float x) const;

    Matrix operator-(const Matrix& B) const;

    Matrix operator-(float x) const;

    Matrix operator==(const Matrix& B) const;

    Matrix operator>(float x) const;

    Matrix operator<(float x) const;

    Matrix broadcastAdd(const Matrix& B) const;

    Matrix broadcastDiv(const Matrix& B) const;

    Matrix T() const;

    static Matrix random(const int rows, const int cols);

    static Matrix randomUniform(const int rows, const int cols, const float start, const float end);

    static Matrix sum(const Matrix& A, int axis = -1);

    static Matrix powM(const Matrix& A, float power);

    static Matrix sqrtM(const Matrix& A);

    static Matrix expM(const Matrix& A);

    static Matrix logM(const Matrix& A);

    static Matrix absM(const Matrix& A);

    static Matrix tanhM(const Matrix& A);

    static Matrix relu(const Matrix& A);

    static Matrix der_relu(const Matrix& A);

    static Matrix maxM(const Matrix& A, int axis = -1);

    static Matrix argmax(const Matrix& A, int axis = 0);

    static Matrix clipM(const Matrix& A, float minValue, float maxValue);

    Matrix clone() const;

    float toScalar() const;
};

Matrix operator*(float x, const Matrix& A);

Matrix operator+(float x, const Matrix& A);

Matrix operator/(float x, const Matrix& A);

Matrix operator-(float x, const Matrix& A);