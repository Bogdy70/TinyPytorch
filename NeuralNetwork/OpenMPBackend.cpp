#include "Backends.h"
#include <stdexcept>
#include <omp.h>
#include <random>

using namespace std;

static random_device rd;
static mt19937 gen(42);

Matrix OpenMPBackend::matmul(const Matrix& A, const Matrix& B)
{
    if(A.getCols() != B.getRows())
    {
        throw runtime_error("Invalid matrix shape for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());

#pragma omp parallel for collapse(2)
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < B.getCols(); j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < A.getCols(); k++)
            {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

Matrix OpenMPBackend::mul(const Matrix& A, const Matrix& B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
    {
        throw runtime_error("Invalid shape for element wise multiplication");
    }

    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) * B(i, j);
        }
    }

    return C;
}

Matrix OpenMPBackend::scalarMul(const Matrix& A, float x)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) * x;
        }
    }

    return C;
}

Matrix OpenMPBackend::div(const Matrix& A, const Matrix& B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
    {
        throw runtime_error("Invalid shape for element wise division");
    }

    Matrix C(A.getRows(), A.getRows());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            if (B(i, j) == 0)
                throw runtime_error("Division by zero");
            C(i, j) = A(i, j) / B(i, j);
        }
    }

    return C;
}

Matrix OpenMPBackend::divScalar(const Matrix& A, float x)
{
    if (x == 0)
        throw runtime_error("Division by zero");

    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) / x;
        }
    }

    return C;
}

Matrix OpenMPBackend::scalarDiv(float x, const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
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

Matrix OpenMPBackend::add(const Matrix& A, const Matrix& B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
    {
        throw runtime_error("Invalid shape for element wise addition");
    }

    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) + B(i, j);
        }
    }

    return C;
}

Matrix OpenMPBackend::scalarAdd(const Matrix& A, float x)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) + x;
        }
    }

    return C;
}

Matrix OpenMPBackend::sub(const Matrix& A, const Matrix& B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
    {
        throw runtime_error("Invalid shape for substraction");
    }

    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) - B(i, j);
        }
    }

    return C;
}

Matrix OpenMPBackend::subScalar(const Matrix& A, float x)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) - x;
        }
    }

    return C;
}

Matrix OpenMPBackend::scalarSub(float x, const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = x - A(i, j);
        }
    }

    return C;
}

Matrix OpenMPBackend::equals(const Matrix& A, const Matrix& B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
        throw runtime_error("Invalid shape");

    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) == B(i, j) ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix OpenMPBackend::greaterth(const Matrix& A, float x)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) > x ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix OpenMPBackend::lessth(const Matrix& A, float x)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) < x ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix OpenMPBackend::broadcastAdd(const Matrix& A, const Matrix& B)
{
    if (A.getRows() != B.getRows() || B.getCols() != 1)
    {
        throw runtime_error("Invalid shape for addition with broadcast");
    }

    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) + B(i, 0);
        }
    }

    return C;
}

Matrix OpenMPBackend::broadcastDiv(const Matrix& A, const Matrix& B)
{
    if (A.getCols() != B.getCols() || B.getRows() != 1)
        throw runtime_error("Invalid shape for division with broadcast");

    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) / B(0, j);
        }
    }

    return C;
}

Matrix OpenMPBackend::T(const Matrix& A)
{
    Matrix Tr(A.getCols(), A.getRows());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            Tr(j, i) = A(i, j);
        }
    }

    return Tr;
}

Matrix OpenMPBackend::random(const int rows, const int cols)
{
    normal_distribution<float> dist(0.0f, 1.0f);

    Matrix C(rows, cols);

#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = dist(gen);
        }
    }

    return C;
}

Matrix OpenMPBackend::randomUniform(const int rows, const int cols, const int start, const int end)
{
    uniform_real_distribution<float> dist(start, end);

    Matrix C(rows, cols);

#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            C(i, j) = dist(gen);
        }
    }

    return C;
}

Matrix OpenMPBackend::sum(const Matrix& A, int axis)
{
    if (axis == 0)
    {
        Matrix C(1, A.getCols());

#pragma omp parallel for
        for (int i = 0; i < A.getCols(); i++)
        {
            float sum = 0.0f;

            for (int j = 0; j < A.getRows(); j++)
            {
                sum += A(j, i);
            }
            C(0, i) = sum;
        }
        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.getRows(), 1);

#pragma omp parallel for
        for (int i = 0; i < A.getRows(); i++)
        {
            float sum = 0.0f;

            for (int j = 0; j < A.getCols(); j++)
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

#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < A.getRows(); i++)
        {
            for (int j = 0; j < A.getCols(); j++)
            {
                sum += A(i, j);
            }
        }
        res(0, 0) = {sum};
        return res;
    }
    else
    {
        throw runtime_error("Invalid axis input. Please input -1, 0 or 1");
    }
}

Matrix OpenMPBackend::powM(const Matrix& A, float power)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = pow(A(i, j), power);
        }
    }

    return C;
}

Matrix OpenMPBackend::sqrtM(const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            if (A(i, j) < 0)
                throw runtime_error("Numbers must not be negative");
            C(i, j) = sqrtf(A(i, j));
        }
    }

    return C;
}

Matrix OpenMPBackend::expM(const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = exp(A(i, j));
        }
    }
    return C;
}

Matrix OpenMPBackend::logM(const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            if (A(i, j) <= 0)
                throw runtime_error("Matrix values must be positive");
            C(i, j) = log(A(i, j));
        }
    }

    return C;
}

Matrix OpenMPBackend::absM(const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = abs(A(i, j));
        }
    }

    return C;
}

Matrix OpenMPBackend::tanhM(const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = tanhf(A(i, j));
        }
    }

    return C;
}

Matrix OpenMPBackend::relu(const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) > 0.0f ? A(i, j) : 0.0f;
        }
    }

    return C;
}

Matrix OpenMPBackend::der_relu(const Matrix& A)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = A(i, j) > 0.0f ? 1.0f : 0.0f;
        }
    }

    return C;
}

Matrix OpenMPBackend::maxM(const Matrix& A, int axis)
{
    if (axis == 0)
    {
        Matrix C(1, A.getCols());

#pragma omp parallel for
        for (int i = 0; i < A.getCols(); i++)
        {
            float maxA = A(0, i);

            for (int j = 1; j < A.getRows(); j++)
            {
                maxA = max(maxA, A(j, i));
            }
            C(0, i) = maxA;
        }

        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.getRows(), 1);

#pragma omp parallel for
        for (int i = 0; i < A.getRows(); i++)
        {
            float maxA = A(i, 0);

            for (int j = 1; j < A.getCols(); j++)
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

#pragma omp parallel for
        for (int i = 0; i < A.getRows(); i++)
        {
            for (int j = 0; j < A.getCols(); j++)
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

Matrix OpenMPBackend::argmax(const Matrix& A, int axis)
{
    if (axis == 0)
    {
        Matrix C(1, A.getCols());

#pragma omp parallel for
        for (int i = 0; i < A.getCols(); i++)
        {
            int idx = 0;
            float maxA = A(0, i);

            for (int j = 1; j < A.getRows(); j++)
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
        Matrix C(A.getRows(), 1);

#pragma omp parallel for
        for (int i = 0; i < A.getRows(); i++)
        {
            int idx = 0;
            float maxA = A(i, 0);

            for (int j = 0; j < A.getCols(); j++)
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

Matrix OpenMPBackend::clipM(const Matrix& A, float minValue, float maxValue)
{
    Matrix C(A.getRows(), A.getCols());

#pragma omp parallel for
    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j < A.getCols(); j++)
        {
            C(i, j) = max(minValue, min(maxValue, A(i, j)));
        }
    }

    return C;
}

Matrix OpenMPBackend::clone(const Matrix& A)
{
    return A;
}

float OpenMPBackend::toScalar(const Matrix& A)
{
    if (A.getRows() != 1 || A.getCols() != 1)
        throw runtime_error("Invalid shape for scalar matrix");

    return A(0, 0);
}