#pragma once

#include "Matrix.h"

#include <cmath>

// Sigmoid Activation Function (Helper)
Matrix<double> sigmoid(const Matrix<double>& in) {
    Matrix<double> out(in.num_rows(), in.num_cols());
    for (size_t i = 0; i < in.num_rows(); ++i) {
        for (size_t j = 0; j < in.num_cols(); ++j) {
            out[i][j] = 1.0 / (1.0 + exp(-in[i][j])); 
        }
    }
    return out;
}


// Softplus Activation Function
Matrix<double> softplus(const Matrix<double>& in) {
    Matrix<double> out(in.num_rows(), in.num_cols());
    for (size_t i = 0; i < in.num_rows(); ++i) {
        for (size_t j = 0; j < in.num_cols(); ++j) {
            out[i][j] = log(exp(in[i][j]) + 1.0); 
        }
    }
    return out;
}

// Derivative of Softplus Activation Function
Matrix<double> dSoftplus(const Matrix<double>& in) {
    Matrix<double> out(in.num_rows(), in.num_cols());
    for (size_t i = 0; i < in.num_rows(); ++i) {
        for (size_t j = 0; j < in.num_cols(); ++j) {
            out[i][j] = 1.0 / (1.0 + exp(-in[i][j]));
        }
    }
    return out;
}

// Second Derivative of activation
Matrix<double> d2Softplus(const Matrix<double>& in) {
    Matrix<double> out(in.num_rows(), in.num_cols());
    for (size_t i = 0; i < in.num_rows(); ++i) {
        for (size_t j = 0; j < in.num_cols(); ++j) {
            double sigmoid = 1.0 / (1.0 + exp(-in[i][j]));
            out[i][j] = sigmoid * (1.0 - sigmoid);
        }
    }
    return out;
}

// MSE calc for Y vector
void MSE_Y(const vector<double>& yTrain, const vector<double>& yPred, double& yMSE) {
    size_t n = yTrain.size();
    yMSE = 0.0;
    for (size_t i = 0; i < n; ++i) {yMSE += ((yTrain[i] - yPred[i]) * (yTrain[i] - yPred[i]));}
    yMSE /= static_cast<double>(n);
}

void MSE_Z(const Matrix<double>& ZTrain, const Matrix<double>& ZPred, double& ZMSE) {
    if (ZTrain.num_rows() != ZPred.num_rows() || ZTrain.num_cols() != ZPred.num_cols()) {throw invalid_argument("Matrix dimensions must match for MSE computation.");}
    size_t m = ZTrain.num_cols(); // number of training points
    // Compute the difference matrix ΔZ
    Matrix<double> DeltaZ = ZPred - ZTrain;

    // Compute the product (ΔZ)^T (ΔZ)
    Matrix<double> M = DeltaZ.transpose()*(DeltaZ);
    double trace = 0.0;
    for (size_t i = 0; i < M.num_rows(); ++i) {trace += M[i][i];}

    // Compute the MSE by dividing the trace by m
    ZMSE = trace / m;
}

// Convert vector to 1D matrix
Matrix<double> vectorToMatrix(const vector<double>& vec) {
    Matrix<double> result(1, vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {result[0][i] = vec[i];}
    return result;
}

// Element-wise division of vectors
vector<double> vec_elementwise_divide(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {result[i] = a[i] / b[i];}
    return result;
}

// Apply a function element-wise to a vector
vector<double> vec_apply(const vector<double>& vec, double (*func)(double)) {
    vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {result[i] = func(vec[i]);}
    return result;
}

// Activations for matrix entries only in BackProp 
double sigmoid(double x) {return 1.0 / (1.0 + exp(-x));}

double dSigmoid(double x) {
    double sigmoid_value = sigmoid(x); 
    return sigmoid_value * (1.0 - sigmoid_value);
}

double softplus(double x) {return log(exp(x) + 1.0);} 

double dSoftplus(double x) {return sigmoid(x);} // Derivative of softplus is sigmoid
double d2Softplus(double x) {
    double sigmoid_value = sigmoid(x);  
    return sigmoid_value * (1.0 - sigmoid_value); 
}

void printIndices(const vector<size_t>& indices) {
    cout << "Indices: ";
    for (size_t i = 0; i < indices.size(); ++i) {
        cout << indices[i];
        if (i != indices.size() - 1) {
            cout << ", ";
        }
    }
    cout << endl;
}