#pragma once

#include "Matrix.h"

struct SequentialBackend
{
    using Mat = Matrix;

    static Mat matmul(const Mat& A, const Mat& B)
    {
        return A.matmul(B);
    }

    static Mat mul(const Mat& A, const Mat& B)
    {
        return A * B;
    }

    static Mat scalarMul(const Mat& A, float x)
    {
        return A * x;
    }

    static Mat div(const Mat& A, const Mat& B)
    {
        return A / B;
    }

    static Mat divScalar(const Mat& A, float x)
    {
        return A / x;
    }

    static Mat scalarDiv(float x, const Mat& A)
    {
        return x / A;
    }

    static Mat add(const Mat& A, const Mat& B)
    {
        return A + B;
    }

    static Mat scalarAdd(const Mat& A, float x)
    {
        return A + x;
    }

    static Mat sub(const Mat& A, const Mat& B)
    {
        return A - B;
    }

    static Mat subScalar(const Mat& A, float x)
    {
        return A - x;
    }

    static Mat scalarSub(float x, const Mat& A)
    {
        return x - A;
    }

    static Mat equals(const Mat& A, const Mat& B)
    {
        return A == B;
    }

    static Mat greaterth(const Mat& A, float x)
    {
        return A > x;
    }

    static Mat lessth(const Mat& A, float x)
    {
        return A < x;
    }

    static Mat broadcastAdd(const Mat& A, const Mat& B)
    {
        return A.broadcastAdd(B);
    }

    static Mat broadcastDiv(const Mat& A, const Mat& B)
    {
        return A.broadcastDiv(B);
    }

    static Mat T(const Mat& A)
    {
        return A.T();
    }

    static Mat random(const int rows, const int cols)
    {
        return Mat::random(rows, cols);
    }

    static Mat randomUniform(const int rows, const int cols, const float start, const float end)
    {
        return Mat::randomUniform(rows, cols, start, end);
    }

    static Mat sum(const Mat& A, int axis = -1)
    {
        return Mat::sum(A, axis);
    }

    static Mat powM(const Mat& A, float power)
    {
        return Mat::powM(A, power);
    }

    static Mat sqrtM(const Mat& A)
    {
        return Mat::sqrtM(A);
    }

    static Mat expM(const Mat& A)
    {
        return Mat::expM(A);
    }

    static Mat logM(const Mat& A)
    {
        return Mat::logM(A);
    }

    static Mat absM(const Mat& A)
    {
        return Mat::absM(A);
    }

    static Mat tanhM(const Mat& A)
    {
        return Mat::tanhM(A);
    }

    static Mat relu(const Mat& A)
    {
        return Mat::relu(A);
    }

    static Mat der_relu(const Mat& A)
    {
        return Mat::der_relu(A);
    }

    static Mat maxM(const Mat& A, int axis = -1)
    {
        return Mat::maxM(A, axis);
    }

    static Mat argmax(const Mat& A, int axis = 0)
    {
        return Mat::argmax(A, axis);
    }

    static Mat clipM(const Mat& A, float minValue, float maxValue)
    {
        return Mat::clipM(A, minValue, maxValue);
    }

    static Mat clone(const Mat& A)
    {
        return A;
    }

    static float toScalar(const Mat& A)
    {
        return A.toScalar();
    }
};

struct OpenMPBackend
{
    using Mat = Matrix;

    static Mat matmul(const Mat& A, const Mat& B);

    static Mat mul(const Mat& A, const Mat& B);

    static Mat scalarMul(const Mat& A, float x);

    static Mat div(const Mat& A, const Mat& B);

    static Mat divScalar(const Mat& A, float x);

    static Mat scalarDiv(float x, const Mat& A);

    static Mat add(const Mat& A, const Mat& B);

    static Mat scalarAdd(const Mat& A, float x);

    static Mat sub(const Mat& A, const Mat& B);

    static Mat subScalar(const Mat& A, float x);

    static Mat scalarSub(float x, const Mat& A);

    static Mat equals(const Mat& A, const Mat& B);

    static Mat greaterth(const Mat& A, float x);

    static Mat lessth(const Mat& A, float x);

    static Mat broadcastAdd(const Mat& A, const Mat& B);

    static Mat broadcastDiv(const Mat& A, const Mat& B);

    static Mat T(const Mat& A);

    static Mat random(const int rows, const int cols);

    static Mat randomUniform(const int rows, const int cols, const int start, const int end);

    static Mat sum(const Mat& A, int axis = -1);

    static Mat powM(const Mat& A, float power);

    static Mat sqrtM(const Mat& A);

    static Mat expM(const Mat& A);

    static Mat logM(const Mat& A);

    static Mat absM(const Mat& A);

    static Mat tanhM(const Mat& A);

    static Mat relu(const Mat& A);

    static Mat der_relu(const Mat& A);

    static Mat maxM(const Mat& A, int axis = -1);

    static Mat argmax(const Mat& A, int axis = 0);

    static Mat clipM(const Mat& A, float minValue, float maxValue);

    static Mat clone(const Mat& A);

    static float toScalar(const Mat& A);
};

struct CUDABackend
{
    using Mat = CMatrix;

    static Mat matmul(const Mat& A, const Mat& B)
    {
        return A.matmul(B);
    }

    static Mat mul(const Mat& A, const Mat& B)
    {
        return A * B;
    }

    static Mat scalarMul(const Mat& A, float x)
    {
        return A * x;
    }

    static Mat div(const Mat& A, const Mat& B)
    {
        return A / B;
    }

    static Mat divScalar(const Mat& A, float x)
    {
        return A / x;
    }

    static Mat add(const Mat& A, const Mat& B)
    {
        return A + B;
    }

    static Mat scalarAdd(const Mat& A, float x)
    {
        return A + x;
    }

    static Mat sub(const Mat& A, const Mat& B)
    {
        return A - B;
    }

    static Mat subScalar(const Mat& A, float x)
    {
        return A - x;
    }

    static Mat equals(const Mat& A, const Mat& B)
    {
        return A == B;
    }

    static Mat greaterth(const Mat& A, float x)
    {
        return A > x;
    }

    static Mat lessth(const Mat& A, float x)
    {
        return A < x;
    }

    static Mat broadcastAdd(const Mat& A, const Mat& B)
    {
        return A.broadcastAdd(B);
    }

    static Mat broadcastDiv(const Mat& A, const Mat& B)
    {
        return A.broadcastDiv(B);
    }

    static Mat T(const Mat& A)
    {
        return A.T();
    }

    static Mat random(const int rows, const int cols)
    {
        return Mat::random(rows, cols);
    }

    static Mat randomUniform(const int rows, const int cols, const float start, const float end)
    {
        return Mat::randomUniform(rows, cols, start, end);
    }

    static Mat sum(const Mat& A, int axis = -1)
    {
        return Mat::sum(A, axis);
    }

    static Mat powM(const Mat& A, float power)
    {
        return Mat::powM(A, power);
    }

    static Mat sqrtM(const Mat& A)
    {
        return Mat::sqrtM(A);
    }

    static Mat expM(const Mat& A)
    {
        return Mat::expM(A);
    }

    static Mat logM(const Mat& A)
    {
        return Mat::logM(A);
    }

    static Mat absM(const Mat& A)
    {
        return Mat::absM(A);
    }

    static Mat tanhM(const Mat& A)
    {
        return Mat::tanhM(A);
    }

    static Mat relu(const Mat& A)
    {
        return Mat::relu(A);
    }

    static Mat der_relu(const Mat& A)
    {
        return Mat::der_relu(A);
    }

    static Mat maxA(const Mat& A, int axis = -1)
    {
        return Mat::maxA(A, axis);
    }

    static Mat argmax(const Mat& A, int axis = 0)
    {
        return Mat::argmax(A, axis);
    }

    static Mat clipM(const Mat& A, float minVal, float maxVal)
    {
        return Mat::clipM(A, minVal, maxVal);
    }

    static Mat scalarDiv(float x, const Mat& A)
    {
        return x / A;
    }

    static Mat scalarSub(float x, const Mat& A)
    {
        return x - A;
    }

    static Mat clone(const Mat& A)
    {
        return A.clone();
    }

    static float toScalar(const Mat& A)
    {
        return A.toScalar();
    }
};