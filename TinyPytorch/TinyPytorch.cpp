#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <stdexcept>
#include "Matrix.h"
#include "CPUTensor.h"

using namespace std;

CMatrix sigmoid(const CMatrix& A)
{
    return 1.0f / (1.0f + CMatrix::expM(-1.0f * A));
}

CMatrix softmax(const CMatrix& A)
{
    CMatrix exp_A = CMatrix::expM(A);
    CMatrix sum_exp = CMatrix::sum(exp_A, 0);

    return exp_A.broadcastDiv(sum_exp);
}

CMatrix der_tanh(const CMatrix& A)
{
    return 1.0f - (CMatrix::tanhM(A) * CMatrix::tanhM(A));
}

CMatrix sign(const CMatrix& A)
{
    return (A > 0.0f) - (A < 0.0f);
}

struct Parameters
{
    vector<CMatrix> W;
    vector<CMatrix> B;

    Parameters(int dim) : W(dim), B(dim) {}
};

float cost(const CMatrix& Y, const CMatrix& pred, const Parameters& params, const float lambda_l1=0.0f, const float lambda_l2=0.0f)
{
    float m = static_cast<float>(Y.getCols());
    int L = size(params.W);
    float epsilon = 1e-8f;
    float cost = 0.0f;
    float sumw = 0.0f;
    CMatrix clippedPred = CMatrix::clipM(pred, epsilon, 1.0f - epsilon);

    if (lambda_l1 < 0.0f || lambda_l2 < 0.0f)
        throw runtime_error("Lambda value cannot be less than zero.");

    if (Y.getRows() > 1)
        cost = (-1.0f / m) * CMatrix::sum(Y * CMatrix::logM(clippedPred)).toScalar();
    else
        cost = (-1.0f / m) * CMatrix::sum(Y * CMatrix::logM(clippedPred) + (1 - Y) * CMatrix::logM(1 - clippedPred)).toScalar();

    if (lambda_l1 > 0.0f)
    {
        sumw = 0.0f;
        for (int l = 1; l < L; l++)
        {
            sumw += CMatrix::sum(CMatrix::absM(params.W[l])).toScalar();
        }
        cost += (lambda_l1 / m) * sumw;
    }
    if (lambda_l2 > 0.0f)
    {
        sumw = 0.0f;
        for (int l = 1; l < L; l++)
        {
            sumw += CMatrix::sum(params.W[l] * params.W[l]).toScalar();
        }
        cost += (lambda_l2 / (2.0f * m)) * sumw;
    }
    return cost;
}

float accuracy(const CMatrix& Y, const CMatrix& pred)
{
    if (Y.getRows() > 1)
    {
        CMatrix true_labels = CMatrix::argmax(Y);
        CMatrix pred_labels = CMatrix::argmax(pred);

        return CMatrix::sum(true_labels == pred_labels).toScalar() / static_cast<float>(Y.getCols());
    }
    else
    {
        CMatrix pred_labels = pred > 0.5f;
        return CMatrix::sum(Y == pred_labels).toScalar() / static_cast<float>(Y.getCols());
    }
}

struct Forward
{
    vector<CMatrix> Z;
    vector<CMatrix> A;
    vector<CMatrix> D;

    Forward(int dim) : Z(dim), A(dim), D(dim){}
};

struct Backward
{
    vector<CMatrix> dZ;
    vector<CMatrix> dW;
    vector<CMatrix> dB;

    Backward(int dim): dZ(dim), dW(dim), dB(dim) {}
};

struct AdamState
{
    vector<CMatrix> VdW;
    vector<CMatrix> VdB;
    vector<CMatrix> SdW;
    vector<CMatrix> SdB;
    int t = 0;

    AdamState(const Parameters& params) : VdW(size(params.W)), VdB(size(params.B)), SdW(size(params.W)), SdB(size(params.B))
    {
        int L = size(params.W);
        for (int l = 1; l < L; l++)
        {
            VdW[l] = CMatrix::zeros(params.W[l].getRows(), params.W[l].getCols());
            VdB[l] = CMatrix::zeros(params.B[l].getRows(), params.B[l].getCols());
            SdW[l] = CMatrix::zeros(params.W[l].getRows(), params.W[l].getCols());
            SdB[l] = CMatrix::zeros(params.B[l].getRows(), params.B[l].getCols());
        }
    }
};

struct Activation
{
    CMatrix(*forward)(const CMatrix&);
    CMatrix(*derivate)(const CMatrix&);

    Activation(): forward(nullptr), derivate(nullptr) {}

    Activation(const string& name)
    {
        if (name == "relu")
        {
            forward = CMatrix::relu;
            derivate = CMatrix::der_relu;
        }
        else if (name == "tanh")
        {
            forward = CMatrix::tanhM;
            derivate = der_tanh;
        }
        else
            throw runtime_error("Invalid activation type. Please choose 'relu' or 'tanh'");
    }
};

struct NetworkConfig
{
    vector<int> dims;
    Activation activation;

    NetworkConfig(const vector<int>& dim_list): dims(dim_list), activation() {}

    NetworkConfig(const vector<int>& dim_list, const string& activ_name) : dims(dim_list), activation(activ_name) {}
};

void printMnistImage(const Matrix& img, int sample_idx)
{
    for (int i = 0; i < 784; i++)
    {
        if (i % 28 == 0)
            cout << "\n";

        float value = img(i, sample_idx);

        if (value > 0.7f)
            cout << "#";
        else if (value > 0.3f)
            cout << ".";
        else
            cout << " ";
    }
}

Parameters init_params(const vector<int>& dim_list)
{
    int L = size(dim_list);
    Parameters params(L);

    for (int l = 1; l < L; l++)
    {
        params.W[l] = CMatrix::random(dim_list[l], dim_list[l - 1]) * sqrt(2.0f / static_cast<float>(dim_list[l - 1]));
        params.B[l] = CMatrix::zeros(dim_list[l], 1);
    }

    return params;
}

Forward forward_pass(const Parameters& params, const CMatrix& X, const string& activation, const float dropout=0.0f)
{
    if (dropout < 0.0f || dropout >= 1.0f)
        throw runtime_error("Dropout value must be between [0, 1).");

    int L = size(params.W);
    Forward forward_cache(L);
    Activation activ(activation);
    CMatrix(*final_activ)(const CMatrix&);

    forward_cache.A[0] = X.clone();

    for (int l = 1; l < L-1; l++)
    {
        forward_cache.Z[l] = params.W[l].matmul(forward_cache.A[l - 1]).broadcastAdd(params.B[l]);
        forward_cache.A[l] = activ.forward(forward_cache.Z[l]);
        if (dropout > 0.0f)
        {
            forward_cache.D[l] = CMatrix::randomUniform(forward_cache.A[l].getRows(), forward_cache.A[l].getCols(), 0.0f, 1.0f) < (1.0f - dropout);
            forward_cache.A[l] = forward_cache.A[l] * forward_cache.D[l] / (1.0f - dropout);
        }
    }
    forward_cache.Z[L - 1] = params.W[L - 1].matmul(forward_cache.A[L - 2]).broadcastAdd(params.B[L - 1]);
    final_activ = forward_cache.Z[L - 1].getRows() > 1 ? softmax : sigmoid;
    forward_cache.A[L - 1] = final_activ(forward_cache.Z[L - 1]);

    return forward_cache;
}

Backward backpropagation(const Forward& frd_cache, const Parameters& params, const CMatrix& Y, const string& activation, const float dropout=0.0f, const float lambda_l1=0.0f, const float lambda_l2=0.0f)
{
    int L = size(frd_cache.A);
    float m = static_cast<float>(Y.getCols());
    Backward grads(L);
    Activation activ(activation);

    grads.dZ[L - 1] = frd_cache.A[L - 1] - Y;
    grads.dW[L - 1] = (1.0f / m) * grads.dZ[L - 1].matmul(frd_cache.A[L - 2].T());
    grads.dB[L - 1] = (1.0f / m) * CMatrix::sum(grads.dZ[L - 1], 1);
    if (lambda_l1 > 0.0f)
        grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l1 / m) * sign(params.W[L - 1]));
    if (lambda_l2 > 0.0f)
        grads.dW[L - 1] = grads.dW[L - 1] + ((lambda_l2 / m) * params.W[L - 1]);

    for (int l = L - 2; l > 0; l--)
    {
        grads.dZ[l] = params.W[l + 1].T().matmul(grads.dZ[l + 1]);
        if (dropout > 0.0f)
            grads.dZ[l] = grads.dZ[l] * frd_cache.D[l] / (1.0f - dropout);
        grads.dZ[l] = grads.dZ[l] * activ.derivate(frd_cache.Z[l]);
        grads.dW[l] = (1.0f / m) * grads.dZ[l].matmul(frd_cache.A[l - 1].T());
        if (lambda_l1 > 0.0f)
            grads.dW[l] = grads.dW[l] + ((lambda_l1 / m) * sign(params.W[l]));
        if (lambda_l2 > 0.0f)
            grads.dW[l] = grads.dW[l] + ((lambda_l2 / m) * params.W[l]);
        grads.dB[l] = (1.0f / m) * CMatrix::sum(grads.dZ[l], 1);
    }

    return grads;
}

Parameters& optimizer(Parameters& params, const Backward& grads, const float lr)
{
    int L = size(params.W);
    for (int l = 1; l < L; l++)
    {
        params.W[l] = params.W[l] - lr * grads.dW[l];
        params.B[l] = params.B[l] - lr * grads.dB[l];
    }

    return params;
}

Parameters& adam(Parameters& params, const Backward& grads, AdamState& state, const float lr, const float beta1, const float beta2, const float epsilon)
{
    if (beta1 < 0.0f || beta1 >= 1.0f)
        throw runtime_error("Beta1 value must be between [0, 1).");
    if (beta2 < 0.0f || beta2 >= 1.0f)
        throw runtime_error("Beta2 value must be between [0, 1).");
    if (epsilon <= 0.0f)
        throw runtime_error("Epsilon must be bigger than 0.");
    int L = size(params.W);
    state.t++;
    float beta1_correction = 1.0f - powf(beta1, state.t);
    float beta2_correction = 1.0f - powf(beta2, state.t);
    for (int l = 1; l < L; l++)
    {
        state.VdW[l] = beta1 * state.VdW[l] + (1.0f - beta1) * grads.dW[l];
        state.VdB[l] = beta1 * state.VdB[l] + (1.0f - beta1) * grads.dB[l];
        state.SdW[l] = beta2 * state.SdW[l] + (1.0f - beta2) * (grads.dW[l] * grads.dW[l]);
        state.SdB[l] = beta2 * state.SdB[l] + (1.0f - beta2) * (grads.dB[l] * grads.dB[l]);

        CMatrix VdW_corrected = state.VdW[l] / beta1_correction;
        CMatrix VdB_corrected = state.VdB[l] / beta1_correction;

        CMatrix SdW_corrected = state.SdW[l] / beta2_correction;
        CMatrix SdB_corrected = state.SdB[l] / beta2_correction;

        params.W[l] = params.W[l] - lr * VdW_corrected / (CMatrix::sqrtM(SdW_corrected) + epsilon);
        params.B[l] = params.B[l] - lr * VdB_corrected / (CMatrix::sqrtM(SdB_corrected) + epsilon);
    }

    return params;
}

Parameters train(const CMatrix& X_train,
    const CMatrix& X_test,
    const CMatrix& y_train,
    const CMatrix& y_test,
    const vector<int>& dim_list,
    const string& activation,
    const float lr,
    const int epochs,
    const int viewing_rate,
    const float beta1 = 0.9f,
    const float beta2 = 0.999f,
    const float eps = 1e-8f,
    const float dropout=0.0f,
    const float lambda_l1=0.0f,
    const float lambda_l2=0.0f)
{
    Parameters params = init_params(dim_list);
    AdamState state(params);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch <= epochs; epoch++)
    {   
        if (epoch % viewing_rate == 0)
        {
            cudaDeviceSynchronize();

            Forward train_eval_cache = forward_pass(params, X_train, activation);
            Forward frd_cache_test = forward_pass(params, X_test, activation);

            float train_objective = cost(y_train, train_eval_cache.A[size(dim_list) - 1], params, lambda_l1, lambda_l2);
            float train_cost = cost(y_train, train_eval_cache.A[size(dim_list) - 1], params);
            float train_acc = accuracy(y_train, train_eval_cache.A[size(dim_list) - 1]);

            float test_cost = cost(y_test, frd_cache_test.A[size(dim_list) - 1], params);
            float test_acc = accuracy(y_test, frd_cache_test.A[size(dim_list) - 1]);

            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;

            cout << "Epoch: " << epoch << " || Train objective: " << train_objective << " || Train loss : " << train_cost << " || Test loss : " << test_cost << " || Train accuracy : " << train_acc * 100.0f << " % || Test accuracy : " << test_acc * 100.0f << " % || Time : " << elapsed.count() << " sec\n";
        }

        if (epoch == epochs)
            break;

        Forward frd_cache = forward_pass(params, X_train, activation, dropout);
        Backward grads = backpropagation(frd_cache, params, y_train, activation, dropout, lambda_l1, lambda_l2);
        //optimizer(params, grads, lr);
        adam(params, grads, state, lr, beta1, beta2, eps);
    }

    return params;
}

void predict(const Matrix& X_test, const Matrix& y_test, const Parameters& params, const string& activation, int imgIdx)
{
    Matrix one_mnist(X_test.getRows(), 1);

    for (int i = 0; i < X_test.getRows(); i++)
    {
        one_mnist(i, 0) = X_test(i, imgIdx);
    }

    Matrix one_mnisty(y_test.getRows(), 1);

    for (int i = 0; i < y_test.getRows(); i++)
    {
        one_mnisty(i, 0) = y_test(i, imgIdx);
    }

    int L = static_cast<int>(params.W.size());

    int pred_label = 0;

    if (y_test.getRows() > 1)
    {
        Forward frd_cache1 = forward_pass(params, one_mnist.toCUDA(), activation);

        pred_label = CMatrix::argmax(frd_cache1.A[L - 1]).toScalar();

        int truth_label = Matrix::argmax(one_mnisty)(0, 0);

        cout << "\nTruth: " << truth_label << " || Pred: " << pred_label;

        printMnistImage(X_test, imgIdx);
    }
    else
    {
        Forward frd_cache1 = forward_pass(params, one_mnist.toCUDA(), activation);

        pred_label = (frd_cache1.A[L - 1] > 0.5f).toCPU()(0, imgIdx);

        cout << "\nTruth: " << y_test(0, imgIdx) << " || Pred: " << pred_label;
    }
}

int main()
{
    try
    {
        Matrix X_train_cat = Matrix::loadMatrixBin("data/cat/X_train.bin", 12288, 209);
        Matrix y_train_cat = Matrix::loadMatrixBin("data/cat/Y_train.bin", 1, 209);

        Matrix X_test_cat = Matrix::loadMatrixBin("data/cat/X_test.bin", 12288, 50);
        Matrix y_test_cat = Matrix::loadMatrixBin("data/cat/Y_test.bin", 1, 50);

        Matrix X_train_mnist = Matrix::loadMatrixBin("data/mnist/X_train.bin", 784, 5000);
        Matrix y_train_mnist = Matrix::loadMatrixBin("data/mnist/Y_train.bin", 10, 5000);

        Matrix X_test_mnist = Matrix::loadMatrixBin("data/mnist/X_test.bin", 784, 1000);
        Matrix y_test_mnist = Matrix::loadMatrixBin("data/mnist/Y_test.bin", 10, 1000);

        cout << "Cat dataset loaded successfully\n";

        cout << "X_train_cat: (" << X_train_cat.getRows() << ", " << X_train_cat.getCols() << ")\n";
        cout << "y_train_cat: (" << y_train_cat.getRows() << ", " << y_train_cat.getCols() << ")\n";
        
        cout << "X_test_cat: (" << X_test_cat.getRows() << ", " << X_test_cat.getCols() << ")\n";
        cout << "y_test_cat: (" << y_test_cat.getRows() << ", " << y_test_cat.getCols() << ")\n";

        cout << "\nMnist dataset loaded successfully\n";

        cout << "X_train_mnist: (" << X_train_mnist.getRows() << ", " << X_train_mnist.getCols() << ")\n";
        cout << "y_train_mnist: (" << y_train_mnist.getRows() << ", " << y_train_mnist.getCols() << ")\n";

        cout << "X_test_mnist: (" << X_test_mnist.getRows() << ", " << X_test_mnist.getCols() << ")\n";
        cout << "y_test_mnist: (" << y_test_mnist.getRows() << ", " << y_test_mnist.getCols() << ")\n\n";


        vector<int> dim_list = { X_train_cat.getRows(), 100, 100, 200, y_train_cat.getRows() };
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed;

        cout << "\n\nTensor test\n\n";

        Tensor T({ 2, 3, 4, 2 });
        T = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 };
        CPUTensor CPUT = T.toCPU();
        CPUT.print();

        cout << "\n\nMatrix test\n\n";

        CMatrix M(4, 2);
        M = { 1, 2, 3, 4, 5, 6, 7, 8 };
        Matrix CPUM = M.toCPU();
        for (int i = 0; i < CPUM.getRows(); i++)
        {
            for (int j = 0; j < CPUM.getCols(); j++)
            {
                cout << CPUM(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nZeros test\n\n";

        Tensor Z = Tensor::zeros({ 2, 3, 3, 3 });
        Z.toCPU().print();


        cout << "\n\nRandom test\n\n";

        CPUTensor R = CPUTensor::random({ 2, 3, 3 });
        R.print();

        cout << "\n";

        CPUTensor RU = CPUTensor::randomUniform({ 1, 3, 3 }, 0.0f, 2.0f);
        RU.print();

        cout << "\n";

        CPUTensor::setSeed(42);
        Tensor RC = Tensor::random({ 1, 3, 3 });
        RC.toCPU().print();

        cout << "\n";
        
        CPUTensor::setSeed(42);
        Tensor RC1 = Tensor::random({ 2, 2, 3 });
        RC1.toCPU().print();

        cout << "\n\nFill test\n\n";

        Tensor F = Tensor::fill({ 3, 3, 2 }, 3.7f);
        F.toCPU().print();

        cout << "\n\nReshape test\n\n";

        Tensor S({ 2, 3, 3 });
        S = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        S.toCPU().print();

        cout << "\n";

        S.reshape({ 3, 2, 3 });
        S.toCPU().print();

        cout << "\n\nResize test\n\n";

        Tensor Rs({ 2, 3, 3 });
        Rs = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        Rs.toCPU().print();

        cout << "\n";

        Rs.resize({ 2, 3, 2, 2 });
        Rs.toCPU().print();

        cout << "\n\nCUDA cat dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        Matrix::setSeed(42);

        Parameters params3 = train(X_train_cat.toCUDA(), X_test_cat.toCUDA(), y_train_cat.toCUDA(), y_test_cat.toCUDA(), dim_list, "tanh", 0.0001f, 700, 100, 0.9f, 0.999f, 1e-8f, 0.2f, 0.0f, 0.01f);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA cat training time: " << elapsed.count() << " seconds\n";

        predict(X_test_cat, y_test_cat, params3, "tanh", 7);


        dim_list = { X_train_mnist.getRows(), 100, 100, 200, y_train_mnist.getRows() };


        cout << "\n\nCUDA mnist dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        Matrix::setSeed(123);

        Parameters params6 = train(X_train_mnist.toCUDA(), X_test_mnist.toCUDA(), y_train_mnist.toCUDA(), y_test_mnist.toCUDA(), dim_list, "relu", 0.0001f, 1000, 100, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f, 0.01f);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA mnist training time: " << elapsed.count() << " seconds\n";
        
        predict(X_test_mnist, y_test_mnist, params6, "relu", 101);
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}