#include <random>
#include "Matrix.h"
#include <cmath>
#include <fstream>

using namespace std;

static random_device rd;
static mt19937 gen(rd());

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r* c, 0.0f) {}

float& Matrix::operator()(int row, int col)
{
    return data[row * cols + col];
}

const float& Matrix::operator()(int row, int col) const
{
    return data[row * cols + col];
}

int Matrix::getRows() const
{
    return rows;
}

int Matrix::getCols() const
{
    return cols;
}

float* Matrix::rawData()
{
    return data.data();
}

const float* Matrix::rawData() const
{
    return data.data();
}

void Matrix::setData(const vector<float>& X)
{
    if (X.size() != rows * cols)
        throw runtime_error("Out of bounds");
    data = X;
}

void Matrix::operator=(const vector<float>& X)
{
    if (X.size() != rows * cols)
        throw runtime_error("Out of bounds!");
    data = X;
}

int Matrix::size() const
{
    return rows * cols;
}

CMatrix Matrix::toCUDA() const
{
    CMatrix C(rows, cols);

    cudaMemcpy(C.rawData(), data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    return C;
}

void Matrix::resize(int new_rows, int new_cols)
{
    rows = new_rows;
    cols = new_cols;
    data.resize(new_rows * new_cols, 0.0f);
}

void Matrix::reshape(int new_rows, int new_cols)
{
    if (new_rows * new_cols != rows * cols)
        throw runtime_error("Total shape must be the same as the previous one");

    rows = new_rows;
    cols = new_cols;
}

Matrix Matrix::matmul(const Matrix& B) const
{
    if (cols != B.rows)
    {
        throw runtime_error("Invalid matrix shape for multiplication");
    }

    Matrix C(rows, B.cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < B.cols; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < cols; k++)
            {
                sum += data[i * cols + k] * B(k, j);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

Matrix Matrix::loadMatrixBin(const string& filepath, int rows, int cols)
{
    Matrix mat(rows, cols);

    ifstream file(filepath, ios::binary);

    if (!file)
    {
        throw runtime_error("File " + filepath + " could not be opened");
    }

    file.read(
        reinterpret_cast<char*>(mat.data.data()),
        rows * cols * sizeof(float)
    );

    if (!file)
    {
        throw runtime_error("Error while reading file" + filepath);
    }

    return mat;
}

Matrix Matrix::operator*(const Matrix& B) const
{
    if (rows != B.rows || cols != B.cols)
    {
        throw runtime_error("Invalid shape for element wise multiplication");
    }

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] * B(i, j);
        }
    }

    return C;
}

Matrix Matrix::operator*(float x) const
{
    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] * x;
        }
    }

    return C;
}

Matrix Matrix::operator/(const Matrix& B) const
{
    if (rows != B.rows || cols != B.cols)
    {
        throw runtime_error("Invalid shape for element wise division");
    }

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (B(i, j) == 0)
                throw runtime_error("Division by zero");
            C(i, j) = data[i * cols + j] / B(i, j);
        }
    }

    return C;
}

Matrix Matrix::operator/(float x) const
{
    if (x == 0)
        throw runtime_error("Division by zero");

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] / x;
        }
    }

    return C;
}

Matrix Matrix::operator+(const Matrix& B) const
{
    if (rows != B.rows || cols != B.cols)
    {
        throw runtime_error("Invalid shape for element wise addition");
    }

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] + B(i, j);
        }
    }

    return C;
}

Matrix Matrix::operator+(float x) const
{
    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] + x;
        }
    }

    return C;
}

Matrix Matrix::operator-(const Matrix& B) const
{
    if (rows != B.rows || cols != B.cols)
    {
        throw runtime_error("Invalid shape for substraction");
    }

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] - B(i, j);
        }
    }

    return C;
}

Matrix Matrix::operator-(float x) const
{
    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] - x;
        }
    }

    return C;
}

Matrix Matrix::operator==(const Matrix& B) const
{
    if (rows != B.rows || cols != B.cols)
        throw runtime_error("Invalid shape");

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] == B(i, j) ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix Matrix::operator>(float x) const
{
    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] > x ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix Matrix::operator<(float x) const
{
    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] < x ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix Matrix::broadcastAdd(const Matrix& B) const
{
    if (rows != B.rows || B.cols != 1)
    {
        throw runtime_error("Invalid shape for addition with broadcast");
    }

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] + B(i, 0);
        }
    }

    return C;
}

Matrix Matrix::broadcastDiv(const Matrix& B) const
{
    if (cols != B.cols || B.rows != 1)
        throw runtime_error("Invalid shape for division with broadcast");

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = data[i * cols + j] / B(0, j);
        }
    }

    return C;
}

Matrix Matrix::T() const
{
    Matrix Tr(cols, rows);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            Tr(j, i) = data[i * cols + j];
        }
    }

    return Tr;
}

void Matrix::setSeed(const int seed)
{
    gen.seed(seed);
}

Matrix Matrix::random(const int rows, const int cols)
{
    normal_distribution<float> dist(0.0f, 1.0f);

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = dist(gen);
        }
    }

    return C;
}

Matrix Matrix::randomUniform(const int rows, const int cols, const float start, const float end)
{
    uniform_real_distribution<float> dist(start, end);

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = dist(gen);
        }
    }

    return C;
}

Matrix Matrix::sum(const Matrix& A, int axis)
{
    if (axis == 0)
    {
        Matrix C(1, A.cols);

        for (int i = 0; i < A.cols; i++)
        {
            float sum = 0.0f;

            for (int j = 0; j < A.rows; j++)
            {
                sum += A(j, i);
            }
            C(0, i) = sum;
        }
        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.rows, 1);

        for (int i = 0; i < A.rows; i++)
        {
            float sum = 0.0f;

            for (int j = 0; j < A.cols; j++)
            {
                sum += A(i, j);
            }
            C(i, 0) = sum;
        }
        return C;
    }
    else if (axis == -1)
    {
        Matrix res(1, 1);
        float sum = 0.0f;

        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                sum += A(i, j);
            }
        }
        res.data = { sum };
        return res;
    }
    else
    {
        throw runtime_error("Invalid axis input. Please input -1, 0 or 1");
    }
}

Matrix Matrix::powM(const Matrix& A, float power)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = pow(A(i, j), power);
        }
    }

    return C;
}

Matrix Matrix::sqrtM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            if (A(i, j) < 0)
                throw runtime_error("Numbers must not be negative");
            C(i, j) = sqrtf(A(i, j));
        }
    }

    return C;
}

Matrix Matrix::expM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = exp(A(i, j));
        }
    }
    return C;
}

Matrix Matrix::logM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            if (A(i, j) <= 0)
                throw runtime_error("Matrix values must be positive");
            C(i, j) = log(A(i, j));
        }
    }

    return C;
}

Matrix Matrix::absM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = abs(A(i, j));
        }
    }

    return C;
}

Matrix Matrix::tanhM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = tanhf(A(i, j));
        }
    }

    return C;
}

Matrix Matrix::relu(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = A(i, j) > 0.0f ? A(i, j) : 0.0f;
        }
    }

    return C;
}

Matrix Matrix::der_relu(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = A(i, j) > 0.0f ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix Matrix::maxM(const Matrix& A, int axis)
{
    if (axis == 0)
    {
        Matrix C(1, A.cols);

        for (int i = 0; i < A.cols; i++)
        {
            float maxA = A(0, i);

            for (int j = 1; j < A.rows; j++)
            {
                maxA = max(maxA, A(j, i));
            }
            C(0, i) = maxA;
        }

        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.rows, 1);

        for (int i = 0; i < A.rows; i++)
        {
            float maxA = A(i, 0);

            for (int j = 1; j < A.cols; j++)
            {
                maxA = max(maxA, A(i, j));
            }
            C(i, 0) = maxA;
        }

        return C;
    }
    else if (axis == -1)
    {
        Matrix C(1, 1);

        float maxA = A(0, 0);

        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                maxA = max(maxA, A(i, j));
            }
        }

        C(0, 0) = maxA;
        return C;
    }
    else
        throw runtime_error("Invalid axis value. Please input -1, 0 or 1");
}

Matrix Matrix::argmax(const Matrix& A, int axis)
{
    if (axis == 0)
    {
        Matrix C(1, A.cols);

        for (int i = 0; i < A.cols; i++)
        {
            int idx = 0;
            float maxA = A(0, i);

            for (int j = 1; j < A.rows; j++)
            {
                if (A(j, i) > maxA)
                {
                    maxA = A(j, i);
                    idx = j;
                }
            }
            C(0, i) = idx;
        }

        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.rows, 1);

        for (int i = 0; i < A.rows; i++)
        {
            int idx = 0;
            float maxA = A(i, 0);

            for (int j = 0; j < A.cols; j++)
            {
                if (A(i, j) > maxA)
                {
                    maxA = A(i, j);
                    idx = j;
                }
            }
            C(i, 0) = idx;
        }

        return C;
    }
    else
        throw runtime_error("Invalid axis false. Please input 0 or 1");
}

Matrix Matrix::clipM(const Matrix& A, float minValue, float maxValue)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = max(minValue, min(maxValue, A(i, j)));
        }
    }

    return C;
}

Matrix Matrix::clone() const
{
    Matrix C(rows, cols);
    C.setData(data);

    return C;
}

float Matrix::toScalar() const
{
    if (rows != 1 || cols != 1)
        throw runtime_error("Invalid shape for scalar matrix");

    return data[0];
}

Matrix operator*(float x, const Matrix& A)
{
    return A * x;
}

Matrix operator+(float x, const Matrix& A)
{
    return A + x;
}

Matrix operator/(float x, const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            if (A(i, j) == 0)
                throw runtime_error("Division by zero");
            C(i, j) = x / A(i, j);
        }
    }

    return C;
}

Matrix operator-(float x, const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = x - A(i, j);
        }
    }

    return C;
}