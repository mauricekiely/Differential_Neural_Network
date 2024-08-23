#pragma once

#include "Matrix.h"
#include "ActivationFuncs.h"
#include "Initialization.h"

#include <thread>  // Include the thread header
#include <chrono>

#include <iostream>

class Layer {
    // Layer Size and Index
    size_t          myNNodes,   myLayerIdx;

    // Matrices for Pre-Activated, Activated and Weight Matrices and their respectove Derivatives
    Matrix<double>  myX,    myZ,    myW;
    vector<double>  myB;

    Matrix<double>  mydYdX, mydYdZ, mydCdW;
    vector<double>  mydCdB;

    // Tensors for updating weights and biases
    Matrix<Matrix<double>> myWTensor;
    vector<Matrix<double>> myBTensor;

public:
    Layer(const vector<size_t>& layerSizes, const size_t layerIdx, const size_t n) :
        myNNodes(layerSizes[layerIdx]),     myLayerIdx(layerIdx),           // Initialize the Node sizes and Indices of Layerss

        myX(layerSizes[layerIdx], n),   myZ(layerSizes[layerIdx], n),     // Initialize the X and Z Matrices of Layers
        myW(layerSizes[layerIdx], layerSizes[layerIdx - 1]),      myB(layerSizes[layerIdx], 0.2),        // Initialize the Weights and Bias Matrices of Layers

        mydYdX(layerSizes[layerIdx], n, (layerIdx == (layerSizes.size() - 1) ? 1.0 : 0.0)),     mydYdZ(layerSizes[layerIdx], n),       // Initalize the Derivatives w.r.t training data   
        mydCdW(layerSizes[layerIdx], layerSizes[layerIdx - 1], 0.0),    mydCdB (layerSizes[layerIdx], 0.0) ,        // Initialize Derivtives w.r.t Weights and Biases    

        myWTensor(layerSizes[layerIdx], layerSizes[layerIdx - 1], Matrix<double>(layerSizes[0], n, 0.0)),  // Initialize the tensor required to update weights w.r.t ∂Y∂X component of cost
        myBTensor(layerSizes[layerIdx], Matrix<double>(layerSizes[0], n, 0.0))  // Initialize the tensor required to update biases w.r.t ∂Y∂X component of cost
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

    Matrix<Matrix<double>>& getWTensor() {return myWTensor;}
    vector<Matrix<double>>& getBTensor() {return myBTensor;}

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
    void backPropagationOfCost(double alpha) {
        // Convert vectors to 1D matrices for computation of ∂MSE_Y/∂W
        Matrix<double> matYPred = vectorToMatrix(myYPred);
        Matrix<double> matYTrain = vectorToMatrix(myYTrain);

        // Initialize the result matrix for the output layer's gradients
        Matrix<double> result = ((matYPred - matYTrain) * (2.0 / myYPred.size())).elementwiseMultiply(dSoftplus(myLayers.back().getZ()));

        // Loop through the layers to backpropagate the gradients
        for (int i = myLayers.size() - 1; i >= 0; --i) {
            if (i > 0) {
                // Calculate the gradient w.r.t. the weights of layer i and accumulate the alpha component
                myLayers[i].getdCdW() = (result.dot(myLayers[i - 1].getdYdX().transpose()) * alpha);
            } else {
                // For the first layer, calculate the gradient w.r.t. the weights using XTrain and accumulate the alpha component
                myLayers[i].getdCdW() = (result.dot(myXTrain.transpose()) * alpha);
            }

            // Calculate the gradient w.r.t. the biases of layer i and accumulate the alpha component
            myLayers[i].getdCdB() = (result.sumAlongRows() * alpha);

            // If not the first layer, propagate the gradient backward
            if (i > 0) {
                result = myLayers[i].getW().transpose().dot(result).elementwiseMultiply(dSoftplus(myLayers[i - 1].getZ()));
            }
        }
    }

    void populateWeightTensors(double beta) {
        size_t n = myZTrain.num_cols();  // Number of training points

        // Iterate over each layer, starting from k=0 (first hidden layer)
        for (size_t k = 0; k < myLayers.size(); ++k) {
            Matrix<double>& dCdW = myLayers[k].getdCdW();

            // Iterate over each weight in the weight matrix W of the current layer
            for (size_t a = 0; a < myLayers[k].getW().num_rows(); ++a) {
                for (size_t b = 0; b < myLayers[k].getW().num_cols(); ++b) {
                    // Use a temporary matrix for calculations instead of storing the full tensor
                    Matrix<double> tempMatrix(myLayerSizes[0], n, 0.0);
                    
                    vector<size_t> indices(myLayerSizes.size() - 1, 0);
                    
                    // Populate the temporary matrix (inputs x n) of ∂C∂W
                    for (size_t i = 0; i < myLayerSizes[0]; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            tempMatrix[i][j] = individualWeightTensorEntry(i, j, a, b, k, indices);
                        }
                    }

                    // Calculate the Frobenius norm of the temporary matrix
                    double norm = 0.0;
                    for (size_t i = 0; i < tempMatrix.num_rows(); ++i) {
                        for (size_t j = 0; j < tempMatrix.num_cols(); ++j) {
                            double diff = tempMatrix[i][j];
                            norm += diff * diff;
                        }
                    }
                    norm = sqrt(norm);  // Take the square root of the sum of squares

                    // Scale by the factor of (2.0 / n) for MSE and multiply by beta
                    norm = beta * (2.0 / n) * norm;

                    // Accumulate the norm into dCdW
                    dCdW[a][b] += norm;
                }
            }
        }
    }

    void populateBiasTensors(double beta) {
        size_t n = myXTrain.num_cols();  // Number of training points

        // Iterate over each layer, starting from k=0 (first hidden layer)
        for (size_t k = 0; k < myLayers.size(); ++k) {
            vector<double>& dCdB = myLayers[k].getdCdB();  // Reference to the condensed vector of norms

            // Iterate over each bias in the bias vector B of the current layer
            for (size_t a = 0; a < myLayers[k].getB().size(); ++a) {
                // Use a temporary matrix for calculations instead of storing the full tensor
                Matrix<double> tempMatrix(myLayerSizes[0], n, 0.0);
                
                vector<size_t> indices(myLayerSizes.size() - 1, 0);
                
                // Populate the temporary matrix (inputs x n) of ∂C∂B
                for (size_t i = 0; i < myLayerSizes[0]; ++i) {
                    for (size_t j = 0; j < myXTrain.num_cols(); ++j) {
                        // Get value for ∂C_{ij}/∂B_a
                        tempMatrix[i][j] = productRuleComponentOfBackProp(i, j, a, 0, k, indices, true) * (myZPred[i][j] - getZTrain()[i][j]);
                    }
                }

                // Calculate the Frobenius norm of the temporary matrix
                double norm = 0.0;
                for (size_t i = 0; i < tempMatrix.num_rows(); ++i) {
                    for (size_t j = 0; j < tempMatrix.num_cols(); ++j) {
                        norm += tempMatrix[i][j] * tempMatrix[i][j];
                    }
                }
                norm = sqrt(norm);  // Take the square root of the sum of squares

                // Scale by the factor of (2.0 / n) for MSE and multiply by beta
                norm = beta * (2.0 / n) * norm;

                // Accumulate the norm into dCdB
                dCdB[a] += norm;
            }
        }
    }

    // Get Derivative w.r.t individual Cost matrix Entry
    double individualWeightTensorEntry(size_t i, size_t j, size_t a, size_t b, size_t k, vector<size_t>& Indices) {
        // Indices for each layer, excluding the input layer
        vector<size_t> indices(myLayerSizes.size() - 1, 0);  
        
        // Product Rule part of Weight derivative
        double productSum = 0.0;
        recursiveProdSum(i, j, a, b, k, myLayerSizes, indices, 0, productSum);
        
        // Trailing part of derivative
        double trailingSum = 0.0;
        if (k == 0) {
            recursiveTrailingTerms(i, j, a, b, k, myLayerSizes, indices, 1, trailingSum);  // Skip the outermost loop
        } else {
            recursiveTrailingTerms(i, j, a, b, k, myLayerSizes, indices, 0, trailingSum);
        }

        // Add part mse component of cost derivative
        double mse_component = (myZPred[i][j] - getZTrain()[i][j]); 
        return (productSum + trailingSum) * mse_component;
    }

    // Recursive function to apply the varying amount of nested summations 
    void recursiveProdSum(size_t i, size_t j, size_t a, size_t b, size_t k, const vector<size_t>& layerSizes, vector<size_t>& indices, size_t currentLayer, double& sum) {
        if (currentLayer == layerSizes.size() - 1) {
            // Base case: All layers processed, compute the product rule component and add to the sum
            double result = productRuleComponentOfBackProp(i, j, a, b, k, indices, false);
            sum += result;
            return;
        }

        // Recursive case: Iterate over the range for the current layer and recurse
        for (size_t n = 0; n < layerSizes[currentLayer + 1]; ++n) {
            indices[currentLayer] = n;  // Update the index for the current layer
            recursiveProdSum(i, j, a, b, k, layerSizes, indices, currentLayer + 1, sum);  // Recur for the next layer
        }
    }

    // Similar to before
   void recursiveTrailingTerms(size_t i, size_t j, size_t a, size_t b, size_t k, const vector<size_t>& layerSizes, vector<size_t>& indices, size_t currentLayer, double& sum) {
        if (currentLayer == layerSizes.size() - 1) {
            // Base case: All layers processed, compute the trailing terms and add to the sum
            double result = trailingTermsOfProd(i, j, a, b, k, indices);
            sum += result;
            return;
        }

        // Handle the special cases where indices are fixed due to W_{ab} Components in derivative
        if (currentLayer == k) {
            recursiveTrailingTerms(i, j, a, b, k, layerSizes, indices, currentLayer + 1, sum);
        } else if (currentLayer == k - 1) {
            recursiveTrailingTerms(i, j, a, b, k, layerSizes, indices, currentLayer + 1, sum);
        } else {
            // Recursive case: Iterate over the range for the current layer and recurse
            for (size_t n = 0; n < layerSizes[currentLayer + 1]; ++n) {
                indices[currentLayer] = n;  // Update the index for the current layer
                recursiveTrailingTerms(i, j, a, b, k, layerSizes, indices, currentLayer + 1, sum);  // Recur for the next layer
            }
        }
    }


    double productRuleComponentOfBackProp(size_t i, size_t j, size_t a, size_t b, size_t k, vector<size_t> Indices, bool isBias) {
        // Find leading weight terms 
        double weightComponent = myLayers[0].getW()[Indices[0]][i];
        for (size_t m = 1; m < Indices.size(); ++m) {weightComponent *= myLayers[m].getW()[Indices[m]][Indices[m-1]];}

        // We need to access Xtrain instead of activated nodes in a layer for k = 0
        Matrix<double>& currX = (k == 0) ? getXTrain() : myLayers[k-1].getX();

        // Initialize Product Rule component for weight or biases
        double prodRuleTotal = (isBias) ? d2Softplus(myLayers[k].getZ()[a][j]) : d2Softplus(myLayers[k].getZ()[a][j]) * currX[b][j];

        // Product of remaining elements of kth part of product rule
        for (size_t m = k + 1; m < getNumLayers() - 1; ++m) {prodRuleTotal *= dSoftplus(myLayers[m].getZ()[Indices[m]][j]);}

        // Sum through rest of the product rule
        for (size_t l = k + 1; l < getNumLayers() - 1; ++l) {
            double tempSum = d2Softplus(myLayers[l].getZ()[Indices[l]][j]);

            // Before l
            for (size_t m = k + 2; m <= l; ++m) {tempSum *= myLayers[m].getW()[Indices[m]][Indices[m-1]] * dSoftplus(myLayers[m-1].getZ()[Indices[m-1]][j]);}

            // Index specific terms of second derivative
            tempSum *= myLayers[k+1].getW()[Indices[k+1]][a] * dSoftplus(myLayers[k].getZ()[a][j]);
            if (!isBias) {tempSum *= currX[b][j];}

            // Remaining terms of product rule
            for (size_t m = k; m < getNumLayers() - 1; ++m) {
                if (m != l) {tempSum *= dSoftplus(myLayers[m].getZ()[Indices[m]][j]);}
            }
            prodRuleTotal += tempSum;
        }

        // Product for all sigma Zs not included in product rule
        for (size_t m = 0; m < k; ++m) {prodRuleTotal *= dSoftplus(myLayers[m].getZ()[Indices[m]][j]);}

        double total = prodRuleTotal * weightComponent;
        return total;
    }

    // Trailing terms from product rule
    double trailingTermsOfProd(size_t i, size_t j, size_t a, size_t b, size_t k, vector<size_t> Indices) {
        Indices[k] = a;
        double trailProdTotal;
        if (k == 0) {
            // Special case for k == 0: only use the first condition. Only have trailing terms for W^(0)_{n i}, i == b
            if (i != b) {return 0.0;}
            trailProdTotal =  dSoftplus(myLayers[k].getZ()[a][j]);
        } else {
            Indices[k-1] = b;
            // General case for k > 0
            trailProdTotal = dSoftplus(myLayers[k-1].getZ()[b][j]) * dSoftplus(myLayers[k].getZ()[a][j]);

            // Multiply the leading terms up to k-2
            for (size_t m = 0; m <= k-2; ++m) {
                trailProdTotal *= dSoftplus(myLayers[m].getZ()[Indices[m]][j]); 
            }
        }

        // Multiply the trailing terms from k+1 onwards
        for (size_t m = k+1; m < getNumLayers() - 1; ++m) {trailProdTotal *= dSoftplus(myLayers[m].getZ()[Indices[m]][j]);}

        // Calculate the weight component, excluding W^{(k)}
        double weightComponent = (k != 0) ? myLayers[0].getW()[Indices[0]][i] : 1.0;
        for (size_t m = 1; m < Indices.size(); ++m) {
            if (m != k) { weightComponent *= myLayers[m].getW()[Indices[m]][Indices[m-1]];}
        }

        return trailProdTotal * weightComponent;
    }

    // Update weights and Biases with Constant Step Size
    void updateWeights(double trainingRate) {
        for (Layer& myLayer:myLayers) {
            myLayer.getW() = myLayer.getW() - (myLayer.getdCdW() * trainingRate);
            myLayer.getB() = myLayer.getB() - (myLayer.getdCdB() * trainingRate);
        }
    }

    // Train for MSEY and MSEZ of training data
    void train(size_t epochs, double trainingRate, double lambda, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, size_t patience = 5, double convergenceThreshold = 0.001) {
        // Get alpha and Beta Values
        double alpha = 1.0 / (1.0 + lambda); 
        double beta = lambda / (1.0 + lambda);

        // Variables for early stopping
        double bestMSEY = std::numeric_limits<double>::infinity();
        size_t epochsSinceImprovement = 0;

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            // Forward Propgate
            forwardPropogation();         
            
            // Compute gradients w.r.t MSE_Y and accumulate them with the alpha component
            backPropagationOfCost(alpha); 
            // Populate the weight tensors, compute gradients w.r.t MSE_Z, and accumulate them with the beta component
            populateWeightTensors(beta);
            populateBiasTensors(beta);

            // Update weights
            updateWeights(trainingRate);

            // Calculate the combined MSE using the specified formula
            double currentMSEY = getMSEY();

            // Output the current epoch's MSE values
            cout << "Epoch " << epoch << ": MSE_Y = " << currentMSEY << endl;

            // Check for early stopping based on MSE_Y alone
            if (currentMSEY < bestMSEY - convergenceThreshold) {
                bestMSEY = currentMSEY;
                epochsSinceImprovement = 0; // Reset counter if improved
            } else {
                epochsSinceImprovement++;
            }

            if (epochsSinceImprovement >= patience) {
                cout << "Early stopping at epoch " << epoch << " as MSE_Y has not improved for " << patience << " consecutive epochs." << endl;
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