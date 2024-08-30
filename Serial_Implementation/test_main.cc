#include <iostream>
#include <chrono>

#include "Network.h"

int main() {
    using namespace std::chrono;

    // Initialize Network Sizes. Note input is 5 for option data
    vector<size_t> layerSizes = {5, 8, 8, 1}; 
    size_t n = 1000;
    double trainRatio = 0.8;

    // Training data 
    Matrix<double> xData(layerSizes[0], n);
    Matrix<double> zData(layerSizes[0], n);
    vector<double> yData(n);

    // Timer for data generation
    auto startDataGen = high_resolution_clock::now();
    // Generate Call Option Data
    generateSyntheticData(layerSizes, n, xData, zData, yData, 1234);
    auto endDataGen = high_resolution_clock::now();
    auto durationDataGen = duration_cast<milliseconds>(endDataGen - startDataGen).count();
    cout << "Time taken for data generation: " << durationDataGen << " ms" << endl;

    size_t trainSize = static_cast<size_t>(n * trainRatio);
    size_t testSize = n - trainSize;

    // Initialize the Matrices required for training
    Matrix<double> xTrain(layerSizes[0], trainSize);
    Matrix<double> zTrain(layerSizes[0], trainSize);
    vector<double> yTrain(trainSize);

    Matrix<double> xTest(layerSizes[0], testSize);
    Matrix<double> zTest(layerSizes[0], testSize);
    vector<double> yTest(testSize);

    // Populate Training Matrices
    splitData(xData, zData, yData, xTrain, xTest, zTrain, zTest, yTrain, yTest, trainRatio, 0.25);

    // Timer for neural network creation
    auto startNNCreation = high_resolution_clock::now();
    // Initialize the neural network with the training data
    NeuralNetwork nn(layerSizes, trainSize, xTrain, zTrain, yTrain, xTest, zTest, yTest);
    auto endNNCreation = high_resolution_clock::now();
    auto durationNNCreation = duration_cast<milliseconds>(endNNCreation - startNNCreation).count();
    cout << "Time taken for NN creation: " << durationNNCreation << " ms" << endl;

    cout << "Number of layers in the network: " << nn.getNumLayers() << endl;

    // Print information for each layer
    cout << "Size of X in each Layer" << endl;
    for (size_t i = 0; i < nn.getNumLayers() - 1; i++) {
        auto dimensions = nn.getLayers()[i].getX().getDimensions();
        cout << "Layer " << i + 1 << ": (" << dimensions.first << ", " << dimensions.second << ")" << endl;
    }

    // Timer for neural network training
    auto startNNTraining = high_resolution_clock::now();
    nn.train(1000, 0.1, 0.0);
    auto endNNTraining = high_resolution_clock::now();
    auto durationNNTraining = duration_cast<milliseconds>(endNNTraining - startNNTraining).count();
    cout << "Time taken for NN training: " << durationNNTraining << " ms" << endl;

    return 0;
}