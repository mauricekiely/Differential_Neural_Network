#pragma once

#include "Matrix.h"
#include "ActivationFuncs.h"
#include "Initialization.h"

#include <iostream>

class Layer {
    // Layer Size and Index
    size_t          myNNodes,   myLayerIdx;

    // Matrices for Pre-Activated, Activated and Weight Matrices and their respectove Derivatives
    Matrix<double>  myX,    myZ,    myW;
    vector<double>  myB;

    Matrix<double>  mydYdX, mydYdZ, mydCdW;
    vector<double>  mydCdB;

    // Vectors and Matrices for Adam Optimiser
    Matrix<double>  myMW,     myVW;
    vector<double>  myMB,     myVB;

public:
    Layer(const vector<size_t>& layerSizes, const size_t layerIdx, const size_t n) :
        myNNodes(layerSizes[layerIdx]),     myLayerIdx(layerIdx),           // Initialize the Node sizes and Indices of Layerss

        myX(layerSizes[layerIdx], n),   myZ(layerSizes[layerIdx], n),     // Initialize the X and Z Matrices of Layers
        myW(layerSizes[layerIdx], layerSizes[layerIdx - 1]),      myB(layerSizes[layerIdx], 0.2),        // Initialize the Weights and Bias Matrices of Layers

        mydYdX(layerSizes[layerIdx], n, (layerIdx == (layerSizes.size() - 1) ? 1.0 : 0.0)),     mydYdZ(layerSizes[layerIdx], n),       // Initalize the Derivatives w.r.t training data   
        mydCdW(layerSizes[layerIdx], layerSizes[layerIdx - 1], 0.0),    mydCdB (layerSizes[layerIdx], 0.0) ,        // Initialize Derivtives w.r.t Weights and Biases    

        myMW(layerSizes[layerIdx], layerSizes[layerIdx - 1], 0.0),    myVW(layerSizes[layerIdx], layerSizes[layerIdx - 1], 0.0),    // Iniialize Weights for Adam 
        myMB(layerSizes[layerIdx], 0.0),       myVB(layerSizes[layerIdx], 0.0)               // Iniialize Bias for Adam 
        {          
            weightInitializer(myW);
            cout << "Layer " << layerIdx << " Created with W size (" << myW.num_rows() << ", " << myW.num_cols() << "), B size " << myB.size() << endl;
        }

    // Accessor Functions
    size_t getNumNodes() const {return myNNodes;}
    size_t getLayerIndex() const {return myLayerIdx;}

    Matrix<double>& getX() {return myX;}
    Matrix<double>& getZ() {return myZ;}
    Matrix<double>& getW() {return myW;}
    vector<double>& getB() {return myB;}

    Matrix<double>& getdYdX() {return mydYdX;}
    Matrix<double>& getdYdZ() {return mydYdZ;}
    Matrix<double>& getdCdW() {return mydCdW;}
    vector<double>& getdCdB() {return mydCdB;}

    vector<double>& getMB() {return myMB;}
    Matrix<double>& getMW() {return myMW;}
    vector<double>& getVB() {return myVB;}
    Matrix<double>& getVW() {return myVW;}
};


class NeuralNetwork {
    // Number of Layers and respective sizes
    size_t              myNLayers;
    vector<size_t>      myLayerSizes;

    // Matrices and Vectors for all relevant training data
    Matrix<double>      myXTrain,   myZTrain,   myZPred, myXTest, myZTest;
    vector<double>      myYTrain,   myYPred, myYTest;

    // Vector of Layers. Core component of Network
    vector<Layer>       myLayers;

    // Hold MSE of trainig data (Not needed, just for debugging really)
    double              YMSE,       ZMSE;

public:
    NeuralNetwork(const vector<size_t>& layerSizes, const size_t n, const Matrix<double>& xTrain, const Matrix<double>& zTrain, const vector<double>& yTrain, 
                                                                    const Matrix<double>& xTest, const Matrix<double>& zTest, const vector<double>& yTest) : 
        // Initilaize the size of the network and the number of Hidden Layers + Output Layer.
        // Note: Here there is no input layer as there is a linear activation to first hidden layer so we just need Xtrain and Ztrain
        myNLayers(layerSizes.size()),               myLayerSizes(layerSizes), 

        // Initialize the training Data and Matrices to hold predictions
        myXTrain(xTrain),           myZTrain(zTrain),        myZPred(zTrain.num_rows(), zTrain.num_cols()),
        myXTest(xTest),     myZTest(zTest),     myYTrain(yTrain),   myYPred(myYTrain.size()),     myYTest(yTest)
        {
            if (myNLayers < 2) {throw invalid_argument("A neural network must have at least an input and an output layer.");}

            // Initialize the layers excluding the input layer as we only need xTrain. Dim is used for weight in 1st hidden Layer
            for (size_t i = 1; i < myNLayers; ++i) {
                myLayers.emplace_back(layerSizes, i, n);
            }
    }

    void forwardPass(const Matrix<double>& xTrain) {
        // Input Layer processing
        // Z_1 = W_1 * X_0 + B_1
        myLayers[0].getZ() = myLayers[0].getW().dot(xTrain) + myLayers[0].getB();
        // X_1 = σ(Z_1)
        myLayers[0].getX() = softplus(myLayers[0].getZ());

        // Forward pass through the remaining layers
        for (size_t i = 1; i < myLayers.size(); ++i) {
            // Z_i = W_i * X_i-1 + B_i
            myLayers[i].getZ() = myLayers[i].getW().dot(myLayers[i - 1].getX()) + myLayers[i].getB();
            // X_i = σ(Z_i)
            myLayers[i].getX() = softplus(myLayers[i].getZ());
        }

        // Transfer to Ypred Vector from output layer of netork
        for (size_t i = 0; i < myYTrain.size(); ++i) {myYPred[i] = myLayers.back().getX()[0][i];}
        MSE_Y(myYTrain, myYPred, YMSE); // Compute MSE.
    }

    // Essentially repeat past step without disturbing the myLayer vector used in computation
    double forwardPassTest(const Matrix<double>& xTest, const vector<double>& yTest) {
        // Initialize temporary variables for Z and X
        Matrix<double> Z = myLayers[0].getW().dot(xTest) + myLayers[0].getB();
        Matrix<double> X = softplus(Z);

        // Forward pass through the remaining layers
        for (size_t i = 1; i < myLayers.size(); ++i) {
            Z = myLayers[i].getW().dot(X) + myLayers[i].getB();
            X = softplus(Z);
        }

        // Compute predictions and store them in a temporary vector
        vector<double> yPred(yTest.size());
        for (size_t i = 0; i < yTest.size(); ++i) {
            yPred[i] = X[0][i];
        }

        // Compute MSE for the test data
        double testMSE = 0.0;
        MSE_Y(yTest, yPred, testMSE);

        // Return the computed test MSE
        return testMSE;
    }
    
    void backPropagationOfY() {
        // Backpropagate through the layers to find ∂Y∂X_0
        for (int i = myLayers.size() - 1; i >= 0; --i) {
            // Compute dYdZ for the current layer
            // ∂X_i/∂Z_i =  σ'(Z_i)
            myLayers[i].getdYdZ() = dSoftplus(myLayers[i].getZ());
            // ∂Y/∂Z_i = ∂Y/∂X_i ⊙ ∂X_i/∂Z_i
            myLayers[i].getdYdZ() = myLayers[i].getdYdZ().elementwiseMultiply(myLayers[i].getdYdX()); // Element-wise multiplication

            // Compute ∂Y/∂X_i until X_train then break
            if (i == 0) { break; }
            myLayers[i - 1].getdYdX() = myLayers[i].getW().transpose().dot(myLayers[i].getdYdZ());
        }

        // Get ZPred = ∂Y/∂XTrain which is input to first hidden layer (myLayers[0])
        myZPred = myLayers[0].getW().transpose().dot(myLayers[0].getdYdZ());

        // Compute MSE for Z
        MSE_Z(myZTrain, myZPred, ZMSE);
    }

    // Combine 2 parts of forward Pass
    void forwardPropogation() {
        forwardPass(myXTrain);
        backPropagationOfY();
    }

    /* 
        Function to find derivatives of Cost function w.r.t the Weights and Biases:
        
        C = 𝛼 MSE_Y + β MSE_Z,      with 𝛼 = 1/(1 + ƛ) and β = ƛ/(1 + ƛ)

        Hence   ∂C/∂W = 𝛼 (∂MSE_Y/∂W) + β (∂MSE_Z/∂W)
                ∂C/∂B = 𝛼 (∂MSE_Y/∂B) + β (∂MSE_Z/∂B)

        This function breaks the backprop into first finding derivatives w.r.t Z and then w.r.t Y
    */
    void backPropagationOfCost(double lambda) {
        double alpha = 1.0 / (1.0 + lambda), beta = lambda / (1.0 + lambda);

        // Initialize the result matrix with ∂C/∂ZPred.     
        // We build on this result Matrix as we go thorugh layers
        Matrix<double> result = (myZPred - myZTrain) * (2.0 / myZPred.num_cols());

        // Add Z component to ∂C/∂W and ∂C/∂B
        for (size_t i = 0; i < myLayers.size(); ++i) {
            myLayers[i].getdCdW() = (result.dot(myLayers[i].getdYdZ().transpose()) * beta).transpose();
            result = myLayers[i].getW().dot(result);
            myLayers[i].getdCdB() = (result.sumAlongRows() * beta);
            result = result.elementwiseMultiply(myLayers[i].getZ());
        }

        // Convert vectos to 1D matrices for computation of ∂MSE_Y/∂W
        Matrix<double> matYPred = vectorToMatrix(myYPred);
        Matrix<double> matYTrain = vectorToMatrix(myYTrain);

        result = ((matYPred - matYTrain) * (2.0 / myYPred.size())).elementwiseMultiply(dSoftplus(myLayers.back().getZ()));

        // Same logic as above to add additional 𝛼 component to weights and biases
        for (size_t i = myLayers.size() - 1; i >= 0; --i) {
            if (i > 0) {
                myLayers[i].getdCdW() = myLayers[i].getdCdW() + (result.dot(myLayers[i - 1].getdYdX().transpose()) * alpha);
            } else {
                myLayers[i].getdCdW() = myLayers[i].getdCdW() + (result.dot(myXTrain.transpose()) * alpha);
            }

            myLayers[i].getdCdB() = myLayers[i].getdCdB() + (result.sumAlongRows() * alpha);

            if (i == 0) { break; }
            result = myLayers[i].getW().transpose().dot(result).elementwiseMultiply(dSoftplus(myLayers[i - 1].getZ()));
        }
    }

    // Update weights and Biases with Constant Step Size
    void updateWeights(double trainingRate) {
        for (Layer& myLayer:myLayers) {
            myLayer.getW() = myLayer.getW() - (myLayer.getdCdW() * trainingRate);
            myLayer.getB() = myLayer.getB() - (myLayer.getdCdB() * trainingRate);
        }
    }

    // Adam Optimiser for updating Weights
    void updateWeightsAdam(double trainingRate, double beta1, double beta2, double epsilon, int t) {
        for (Layer& myLayer : myLayers) {
            // Update first moment estimate (mW and mB)
            myLayer.getMW() = (myLayer.getdCdW() * (1.0 - beta1)) + (myLayer.getMW() * beta1);
            myLayer.getMB() = (myLayer.getMB() * beta1) +(myLayer.getdCdB()* (1.0 - beta1));

            // Update second moment estimate (vW and vB)
            myLayer.getVW() = (myLayer.getVW() * beta2) + (myLayer.getdCdW().elementwiseMultiply(myLayer.getdCdW()) * (1.0 - beta2));
            myLayer.getVB() = (myLayer.getVB() * beta2) + ((myLayer.getdCdB() * myLayer.getdCdB()) * (1.0 - beta2));

            // Compute bias-corrected first and second moment estimates
            Matrix<double> mW_hat = myLayer.getMW() / (1.0 - pow(beta1, t));
            vector<double> mB_hat = myLayer.getMB() * (1.0 / (1.0 - pow(beta1, t)));

            Matrix<double> vW_hat = myLayer.getVW() / (1.0 - pow(beta2, t));
            vector<double> vB_hat = myLayer.getVB() * (1.0 / (1.0 - pow(beta2, t)));

            // Update weights and biases
            myLayer.getW() = myLayer.getW() - (mW_hat.elementwiseDivide(vW_hat.apply(sqrt) + epsilon) * trainingRate);
            myLayer.getB() = myLayer.getB() - ((mB_hat / (vec_apply(vB_hat, sqrt) + epsilon)) * trainingRate);
        }
    }

    void trainOnYMSE(size_t epochs, double trainingRate, double lambda, size_t patience = 10, double beta1 = 0.9, double beta2 = 0.99, double epsilon = 1e-8) {
        // Initialise best MSE for early CallBack
        double bestMSE_Y = numeric_limits<double>::infinity();
        size_t epochsSinceImprovement = 0;

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            forwardPropogation();         
            backPropagationOfCost(lambda); 

            // Update weights using Adam optimizer
            updateWeightsAdam(trainingRate, beta1, beta2, epsilon, epoch);

            double currentMSE_Y = forwardPassTest(myXTest, myYTest);

            // Check if the current MSE_Y is lower than the best observed so far
            if (currentMSE_Y < bestMSE_Y) {
                bestMSE_Y = currentMSE_Y;
                epochsSinceImprovement = 0; // Reset counter if improvement is found
            } else {epochsSinceImprovement++;}

            // Print progress every 5 epochs
            if (epoch % 5 == 0) {cout << "Epoch " << epoch << ": MSE_Y = " << getMSEY() << ", MSE_Z = " << getMSEZ() << endl;}

            // Early stopping condition
            if (epochsSinceImprovement >= patience) {
                cout << "Early stopping at epoch " << epoch << " as MSE_Y has not improved for " << patience << " consecutive epochs." << endl;
                break;
            }
        }
    }

    void train(size_t epochs, double trainingRate, double lambda, size_t patience = 10, double beta1 = 0.9, double beta2 = 0.99, double epsilon = 1e-8) {
        double bestCombinedMSE = std::numeric_limits<double>::infinity();
        size_t epochsSinceImprovement = 0;

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            forwardPropogation();         // Perform forward propagation
            backPropagationOfCost(lambda); // Compute gradients

            // Update weights using Adam optimizer
            updateWeightsAdam(trainingRate, beta1, beta2, epsilon, epoch);

            // Calculate the combined MSE using the specified formula
            double combinedMSE = ((1.0 / (1.0 + lambda)) * getMSEY()) + ((lambda / (1.0 + lambda)) * getMSEZ());

            // Check if the current combined MSE is lower than the best observed so far
            if (combinedMSE < bestCombinedMSE) {
                bestCombinedMSE = combinedMSE;
                epochsSinceImprovement = 0; // Reset counter if improvement is found
            } else {
                epochsSinceImprovement++;
            }

            // Optionally, print progress every 5 epochs
            if (epoch % 5 == 0) {
                cout << "Epoch " << epoch << ": MSE_Y = " << getMSEY() << ", MSE_Z = " << getMSEZ() << ", Combined MSE = " << combinedMSE << endl;
            }

            // Early stopping condition
            if (epochsSinceImprovement >= patience) {
                cout << "Early stopping at epoch " << epoch << " as combined MSE has not improved for " << patience << " consecutive epochs." << endl;
                break;
            }
        }
    }
    // Accessor methods
    size_t getNumLayers() const {return myNLayers;}   // Node the Layer vector actually has 1 less cos of no input layer object
    const vector<size_t>& getLayerSizes() const {return myLayerSizes;}
    Matrix<double>& getXTrain() {return myXTrain;}
    Matrix<double>& getZTrain() {return myZTrain;}
    vector<double>& getYTrain() {return myYTrain;}
    vector<Layer>& getLayers() {return myLayers;}

    const double& getMSEY() const {return YMSE;}
    const double& getMSEZ() const {return ZMSE;}

    Matrix<double>& getXTest() {return myXTest;}
    Matrix<double>& getZTest() {return myZTest;}
    vector<double>& getYTest() {return myYTest;}
};

// test change