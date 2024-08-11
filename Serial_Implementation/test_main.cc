#include <iostream>
#include "Network.h"

int main() {
    vector<size_t> layerSizes = {5, 4, 3, 1}; 
    size_t n = 1000;
    double trainRatio = 0.75;

    // Example training data 
    Matrix<double> xData(layerSizes[0], n);
    Matrix<double> zData(layerSizes[0], n);
    vector<double> yData(n);

    generateSyntheticData(layerSizes, n, xData, zData, yData, 0.2);

    size_t trainSize = static_cast<size_t>(n * trainRatio);
    size_t testSize = n - trainSize;

    Matrix<double> xTrain(layerSizes[0], trainSize);
    Matrix<double> zTrain(layerSizes[0], trainSize);
    vector<double> yTrain(trainSize);

    Matrix<double> xTest(layerSizes[0], testSize);
    Matrix<double> zTest(layerSizes[0], testSize);
    vector<double> yTest(testSize);

    splitData(xData, zData, yData, xTrain, xTest, zTrain, zTest, yTrain, yTest, trainRatio);

    // Initialize the neural network with the training data
    NeuralNetwork nn(layerSizes, trainSize, xTrain, zTrain, yTrain, xTest, zTest, yTest);

    cout << "Number of layers in the network: " << nn.getNumLayers() << endl;

    // Print information for each layer
    cout << "Size of X in each Layer" << endl;
    for (size_t i = 0; i < nn.getNumLayers() - 1; i++) {
        auto dimensions = nn.getLayers()[i].getX().getDimensions();
        cout << "Layer " << i + 1 << ": (" << dimensions.first << ", " << dimensions.second << ")" << endl;
    }

    nn.trainOnYMSE(1000, 0.001, 1);

    return 0;
}
