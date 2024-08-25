#include <iostream>
#include "Network.h"


int main() {
    // Initialize Network Sizes. Note input is 5 for option data
    vector<size_t> layerSizes = {5, 10, 10, 1}; 
    size_t n = 1000;
    double trainRatio = 0.8;

    // Training data 
    Matrix<double> xData(layerSizes[0], n);
    Matrix<double> zData(layerSizes[0], n);
    vector<double> yData(n);

    // Generate Call Option Data
    generateSyntheticData(layerSizes, n, xData, zData, yData, 1234);


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
    splitData(xData, zData, yData, xTrain, xTest, zTrain, zTest, yTrain, yTest, trainRatio, 0.0);

    // Initialize the neural network with the training data
    NeuralNetwork nn(layerSizes, trainSize, xTrain, zTrain, yTrain, xTest, zTest, yTest);

    cout << "Number of layers in the network: " << nn.getNumLayers() << endl;

    // Print information for each layer
    cout << "Size of X in each Layer" << endl;
    for (size_t i = 0; i < nn.getNumLayers() - 1; i++) {
        auto dimensions = nn.getLayers()[i].getX().getDimensions();
        cout << "Layer " << i + 1 << ": (" << dimensions.first << ", " << dimensions.second << ")" << endl;
    }

    // Train Model for 300 Epochs with Early Stopping to avoid overfitting
    nn.train(200, 0.1, 1.0);

    return 0;
}
