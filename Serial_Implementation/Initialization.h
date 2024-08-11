#pragma once

#include "Matrix.h"

#include <random>

void weightInitializer(Matrix<double>& mat) {
    random_device rd;
    mt19937 gen(1234);
    uniform_real_distribution<> uDist(-1.0, 1.0);

    // Calculate the limit for Xavier initialization
    double limit = sqrt(6.0 / (mat.num_rows() + mat.num_cols()));

    for (size_t i = 0; i < mat.num_rows(); ++i) {
        for (size_t j = 0; j < mat.num_cols(); ++j) {
            mat[i][j] = uDist(gen) * limit;
        }}
}

// Add random nise to a value
double addNoise(double value, double noise_level, mt19937& gen) {
    normal_distribution<> d(0, noise_level);
    return value + d(gen);
}

// Function to generate synthetic data
void generateSyntheticData(vector<size_t>& layerSizes, size_t n, Matrix<double>& xTrain, Matrix<double>& zTrain, vector<double>& yTrain, double noise_level) {
    random_device rd;  // Seed for the random number engine
    mt19937 gen(rd()); // Mersenne Twister random number generator
    uniform_real_distribution<> dis(0.0, 1.0); // Uniform distribution between 0 and 1

    // Non-linear function: y = sin(x1 + x2 + ... + xm)
    // Derivative: dy/dx_i = cos(x1 + x2 + ... + xm) for each x_i

    // Fill xTrain with random values
    for (size_t i = 0; i < xTrain.num_rows(); ++i) {
        for (size_t j = 0; j < xTrain.num_cols(); ++j) {
            xTrain[i][j] = dis(gen);
        }
    }

    // Calculate yTrain and zTrain with noise
    for (size_t j = 0; j < n; ++j) {
        double x_sum = 0.0;
        for (size_t i = 0; i < xTrain.num_rows(); ++i) {x_sum += xTrain[i][j];}

        // Calculate the function value with noise
        double y_value = sin(x_sum);
        yTrain[j] = addNoise(y_value, noise_level, gen);

        // Calculate the derivatives with noise
        double dy_dx = cos(x_sum);
        for (size_t i = 0; i < xTrain.num_rows(); ++i) {zTrain[i][j] = addNoise(dy_dx, noise_level, gen);}
    }
}

// Split data into test and training with proportion trainRatio
void splitData(const Matrix<double>& x, const Matrix<double>& z, const vector<double>& y,
               Matrix<double>& xTrain, Matrix<double>& xTest,
               Matrix<double>& zTrain, Matrix<double>& zTest,
               vector<double>& yTrain, vector<double>& yTest,
               double trainRatio=0.8) {
    size_t n = y.size();
    size_t trainSize = static_cast<size_t>(n * trainRatio);

    for (size_t i = 0; i < trainSize; ++i) {
        for (size_t j = 0; j < x.num_rows(); ++j) {
            xTrain[j][i] = x[j][i];
            zTrain[j][i] = z[j][i];
        }
        yTrain[i] = y[i];
    }

    for (size_t i = trainSize; i < n; ++i) {
        for (size_t j = 0; j < x.num_rows(); ++j) {
            xTest[j][i - trainSize] = x[j][i];
            zTest[j][i - trainSize] = z[j][i];
        }
        yTest[i - trainSize] = y[i];
    }
}