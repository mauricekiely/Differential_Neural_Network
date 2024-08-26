#pragma once

#include "Matrix.h"
#include <random>

// Function to initialize weights using Xavier initialization
void weightInitializer(Matrix<double>& mat) {
    std::random_device rd;
    std::mt19937 gen(1234);
    std::uniform_real_distribution<> uDist(-1.0, 1.0);

    // Calculate the limit for Xavier initialization
    double limit = std::sqrt(6.0 / (mat.num_rows() + mat.num_cols()));

    for (size_t i = 0; i < mat.num_rows(); ++i) {
        for (size_t j = 0; j < mat.num_cols(); ++j) {
            mat[i][j] = uDist(gen) * limit;
        }
    }
}

// Helper function for the CDF of the standard normal distribution
static double N(double z) {
    return 0.5 * (1 + std::erf(z * M_SQRT1_2));
}

// Function to calculate the price and Greeks for a European call option
void calcOptionAndGreeks(double S, double K, double t, double sig, double r, double& call, double& delta, double& vega, double& rho, double& theta) {
    double sig_sqrt_t = sig * std::sqrt(t);
    double d1 = (std::log(S / K) + (r + sig * sig / 2.0) * t) / sig_sqrt_t;
    double d2 = d1 - sig_sqrt_t;

    double Nd1 = N(d1);
    double Nd2 = N(d2);
    double expRT = std::exp(-r * t);

    // Option call price
    call = S * Nd1 - K * expRT * Nd2;

    // Sensitivities (Greeks)
    double dNd1 = std::exp(-d1 * d1 / 2.0) / std::sqrt(2 * M_PI);
    delta = Nd1;
    vega = S * std::sqrt(t) * dNd1;
    rho = K * t * expRT * Nd2;
    theta = (1.0 / t) * (S * sig * dNd1 / (2 * std::sqrt(t)) + r * K * expRT * Nd2);
}

// Add random noise to a value
double addNoise(double value, double noise_level, std::mt19937& gen) {
    std::normal_distribution<> d(0, noise_level);
    return value + d(gen);
}

// Function to standardize a matrix
void standardizeMatrix(Matrix<double>& mat, std::vector<double>& means, std::vector<double>& stdDevs) {
    size_t rows = mat.num_rows();
    size_t cols = mat.num_cols();

    // Calculate mean and standard deviation for each row (feature)
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum += mat[i][j];
        }
        means[i] = sum / cols;

        double sq_sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sq_sum += std::pow(mat[i][j] - means[i], 2);
        }
        stdDevs[i] = std::sqrt(sq_sum / cols);

        // Standardize the data
        for (size_t j = 0; j < cols; ++j) {
            mat[i][j] = (mat[i][j] - means[i]) / stdDevs[i];
        }
    }
}

// Function to standardize a vector
void standardizeVector(std::vector<double>& vec, double& mean, double& stdDev) {
    size_t n = vec.size();

    // Calculate mean and standard deviation
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += vec[i];
    }
    mean = sum / n;

    double sq_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sq_sum += std::pow(vec[i] - mean, 2);
    }
    stdDev = std::sqrt(sq_sum / n);

    // Standardize the data
    for (size_t i = 0; i < n; ++i) {
        vec[i] = (vec[i] - mean) / stdDev;
    }
}

// Function to generate synthetic data based on a European option model
void generateSyntheticData(std::vector<size_t>& layerSizes, size_t n, Matrix<double>& xTrain, Matrix<double>& zTrain, std::vector<double>& yTrain, size_t seed) {
    std::random_device rd;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> spotDist(90.0, 110.0);
    std::uniform_real_distribution<> strikeDist(95.0, 105.0);
    std::uniform_real_distribution<> timeDist(0.1, 3.0);
    std::uniform_real_distribution<> volDist(0.05, 0.30);
    std::uniform_real_distribution<> rateDist(0.001, 0.05);

    for (size_t j = 0; j < n; ++j) {
        // Generate random inputs
        double S = spotDist(gen);
        double K = strikeDist(gen);
        double t = timeDist(gen);
        double sig = volDist(gen);
        double r = rateDist(gen);

        // Calculate the option price and Greeks
        double call, delta, vega, rho, theta;
        calcOptionAndGreeks(S, K, t, sig, r, call, delta, vega, rho, theta);

        // Assign input values to xTrain
        xTrain[0][j] = S;
        xTrain[1][j] = K;
        xTrain[2][j] = t;
        xTrain[3][j] = sig;
        xTrain[4][j] = r;

        // Assign output values to yTrain and zTrain without noise
        yTrain[j] = call;
        zTrain[0][j] = delta;
        zTrain[1][j] = delta;  // Derivative w.r.t. Strike (same as delta)
        zTrain[2][j] = vega;
        zTrain[3][j] = rho;
        zTrain[4][j] = theta;
    }

    // Standardize xTrain, yTrain, and zTrain
    std::vector<double> xMeans(layerSizes[0]);
    std::vector<double> xStdDevs(layerSizes[0]);
    standardizeMatrix(xTrain, xMeans, xStdDevs);

    std::vector<double> zMeans(layerSizes[0]);
    std::vector<double> zStdDevs(layerSizes[0]);
    standardizeMatrix(zTrain, zMeans, zStdDevs);

    double yMean, yStdDev;
    standardizeVector(yTrain, yMean, yStdDev);
}

// Split data into training and test sets and apply standardization
void splitData(const Matrix<double>& x, const Matrix<double>& z, const std::vector<double>& y, Matrix<double>& xTrain, Matrix<double>& xTest, Matrix<double>& zTrain, Matrix<double>& zTest,
               std::vector<double>& yTrain, std::vector<double>& yTest, double trainRatio, double noise_level) {
    size_t n = y.size();
    size_t trainSize = static_cast<size_t>(n * trainRatio);

    // Prepare containers for the test data statistics
    std::vector<double> xMeans(x.num_rows(), 0.0);
    std::vector<double> xStdDevs(x.num_rows(), 0.0);
    std::vector<double> zMeans(z.num_rows(), 0.0);
    std::vector<double> zStdDevs(z.num_rows(), 0.0);
    double yMean = 0.0, yStdDev = 0.0;

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

    // Standardize the training data
    standardizeMatrix(xTrain, xMeans, xStdDevs);
    standardizeMatrix(zTrain, zMeans, zStdDevs);
    standardizeVector(yTrain, yMean, yStdDev);

    // Apply the same standardization to the test data using the training data's mean and std deviation
    for (size_t i = 0; i < xTest.num_rows(); ++i) {
        for (size_t j = 0; j < xTest.num_cols(); ++j) {
            xTest[i][j] = (xTest[i][j] - xMeans[i]) / xStdDevs[i];
        }
    }

    for (size_t i = 0; i < zTest.num_rows(); ++i) {
        for (size_t j = 0; j < zTest.num_cols(); ++j) {
            zTest[i][j] = (zTest[i][j] - zMeans[i]) / zStdDevs[i];
        }
    }
    for (size_t i = 0; i < yTest.size(); ++i) {
        yTest[i] = (yTest[i] - yMean) / yStdDev;
    }

    // Add noise after standardization
    std::random_device rd;
    std::mt19937 gen(rd());  // Use random seed for test set noise
    for (size_t i = 0; i < yTest.size(); ++i) {
        yTest[i] = addNoise(yTest[i], noise_level, gen);
    }
    for (size_t i = 0; i < zTest.num_rows(); ++i) {
        for (size_t j = 0; j < zTest.num_cols(); ++j) {
            zTest[i][j] = addNoise(zTest[i][j], noise_level, gen);
        }
    }
}
